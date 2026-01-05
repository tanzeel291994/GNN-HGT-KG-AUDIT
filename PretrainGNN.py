import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
from torch_geometric.nn.models import SGFormer
from torch_geometric.data import HeteroData
import copy
from tqdm import tqdm
import time
from torch_geometric.loader import HGTLoader  
#from torch_geometric.loader import NeighborLoader


# --- 1. MODEL DEFINITION ---
class PretrainableHeteroGNN(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Local HGT Layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_dim, hidden_dim, metadata, heads=4)) # 4096 x 4096
        self.entity_norm = nn.LayerNorm(hidden_dim)
        self.alpha = 0.1
        # 2. Global SGFormer Layer (SGF)
        self.entity_global_reasoner = SGFormer( # from 4096 to 256 is it reliable ???
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            trans_num_layers=1,
            trans_num_heads=4,
            gnn_num_layers=0,      # only global transformer layer is activated
            graph_weight=0.0       # only global transformer layer is activated
        )
        
        # 3. Projection Head for Contrastive Learning # why all projection head look like this ???
        # [FIX] Separate Projectors for distinct semantic spaces
        self.entity_projector = nn.Sequential(
            nn.Linear(hidden_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
        self.passage_projector = nn.Sequential(
            nn.Linear(hidden_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x_dict, edge_index_dict, batch_dict=None, return_h=False):
        # 1. Message Passing (Update ALL node types)
        h_dict = x_dict
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {k: F.gelu(v) for k, v in h_dict.items()} 
        
        # 2. Entity Reasoning (Same as before)
        h_local_ent = h_dict['entity']
        batch_ent = h_local_ent.new_zeros(h_local_ent.size(0), dtype=torch.long)
        h_ent_global = self.entity_global_reasoner.trans_conv(h_local_ent, batch_ent)
        h_dict['entity'] = self.entity_norm((1 - self.alpha) * h_local_ent + self.alpha * h_ent_global)

        if return_h:
            return h_dict # Return dict to access both entity and passage later

        # [FIX] Project BOTH types
        # We perform contrastive learning on both critical node types
        z_entity = self.entity_projector(h_dict['entity'])
        
        # Check if batch contains passages (it might not if using HGTLoader strictly)
        if 'passage' in h_dict and h_dict['passage'].size(0) > 0:
            z_passage = self.passage_projector(h_dict['passage'])
            return {'entity': z_entity, 'passage': z_passage}
            
        return {'entity': z_entity, 'passage': None}

# --- 2. DATA AUGMENTATION ---
def get_graph_augmentation(data: HeteroData, node_mask_rate=0.1, edge_drop_rate=0.1):
    """
    Creates an augmented version of the HeteroData graph for self-supervised learning.
    """
    aug_data = data.clone()
    
    # Node Feature Masking
    for node_type in aug_data.node_types:
        x = aug_data[node_type].x
        if x is not None:
            mask = torch.rand(x.size(0), device=x.device) < node_mask_rate
            x_aug = x.clone()
            x_aug[mask] = 0
            aug_data[node_type].x = x_aug
            
    # Edge Dropping
    for edge_type in aug_data.edge_types:
        edge_index = aug_data[edge_type].edge_index
        num_edges = edge_index.size(1)
        if num_edges > 0:
            keep_mask = torch.rand(num_edges, device=edge_index.device) > edge_drop_rate
            aug_data[edge_type].edge_index = edge_index[:, keep_mask]
            if 'edge_attr' in aug_data[edge_type] and aug_data[edge_type].edge_attr is not None:
                aug_data[edge_type].edge_attr = aug_data[edge_type].edge_attr[keep_mask]
                
    return aug_data

# --- 3. CONTRASTIVE LOSS (NT-Xent) ---
def contrastive_loss(z1, z2, temperature=0.1, max_nodes=5000):
    """
    NT-Xent loss between two sets of embeddings.
    Samples nodes if the batch is too large to fit in memory.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Subsampling to avoid memory overflow for large graphs
    if z1.size(0) > max_nodes:
        indices = torch.randperm(z1.size(0))[:max_nodes]
        z1 = z1[indices]
        z2 = z2[indices]
    
    batch_size = z1.size(0)
    sim_matrix = torch.mm(z1, z2.t()) / temperature # z1 n x d and z2 n x d
    labels = torch.arange(batch_size, device=z1.device) # n x n 
    return F.cross_entropy(sim_matrix, labels)


# ... (PretrainableHeteroGNN and get_graph_augmentation remain same) ...

# --- 4. PRE-TRAINING LOOP ---
def pretrain(graph_path, hidden_dim, out_dim, epochs=10, lr=1e-3, device='cuda', batch_size=1024):
    print(f"Loading graph to CPU...")
    data = torch.load(graph_path, weights_only=False, map_location='cpu')
    
    # Use standard float32 for compatibility with pyg-lib kernels
    for node_type in data.node_types:
        if data[node_type].x is not None:
            data[node_type].x = data[node_type].x.to(torch.float32) # why this is needed?
            
    # Initialize Model and move to device
    model = PretrainableHeteroGNN(data.metadata(), hidden_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create a HGTLoader for the 'entity' nodes
    # NOTE: In HGTLoader, num_samples is the TOTAL budget of nodes per type per hop for the ENTIRE batch.
    # [10, 5] was way too small for a batch_size of 1024, leaving nodes isolated.
    train_loader = HGTLoader(
        data,
        num_samples=[2048, 1024], # Increased budget to provide structural context
        batch_size=batch_size,
        input_nodes='passage',
        shuffle=True,
        num_workers=8,
        persistent_workers=True
    )
    print(f"Starting Sub-graph Pre-training...")
    model.train()
    
    # Pre-clear cache
    torch.cuda.empty_cache()
    
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # [FIX 1] Get batch size from 'passage' (the new input node)
            curr_batch_size = batch['passage'].batch_size
            
            # Generate views
            view1 = get_graph_augmentation(batch).to(device)
            view2 = get_graph_augmentation(batch).to(device)
            
            # Run Model
            out1 = model(view1.x_dict, view1.edge_index_dict)
            out2 = model(view2.x_dict, view2.edge_index_dict)
            
            # [FIX 2] Calculate Loss on Passages (Primary Objective)
            # We slice [:curr_batch_size] to only train on the "Seed" passages, 
            # ignoring the neighbor nodes pulled in for context.
            z1_passage = out1['passage'][:curr_batch_size]
            z2_passage = out2['passage'][:curr_batch_size]
            
            loss_passage = contrastive_loss(z1_passage, z2_passage)
            
            # [OPTIONAL] Add Entity Loss if you want to keep entity reasoning strong
            # We use ALL entities in the subgraph since they don't have a specific batch order
            # or we can skip it to focus purely on passages.
            if out1['entity'] is not None:
                # Simple full-graph contrastive for entities (lighter weight)
                z1_ent = out1['entity']
                z2_ent = out2['entity']
                # Downsample if too many entities to save memory
                if z1_ent.size(0) > 2000:
                    idx = torch.randperm(z1_ent.size(0))[:2000]
                    z1_ent = z1_ent[idx]
                    z2_ent = z2_ent[idx]
                
                loss_entity = contrastive_loss(z1_ent, z2_ent)
                loss = loss_passage + (0.1 * loss_entity) # Small weight for entities
            else:
                loss = loss_passage
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Aggressive cleanup for large embeddings
            #del view1, view2, z1, z2, batch
            #if num_batches % 20 == 0:
            #    torch.cuda.empty_cache()

        duration = time.time() - start_time
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1:02d} | Avg Loss: {avg_loss:.4f} | Time: {duration:.2f}s")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"pretrained_gnn_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "pretrained_gnn_final.pth")
    print("Pre-training completed and model saved.")

if __name__ == "__main__":
    #GRAPH_PATH = "/home/ubuntu/gnn-hgt/GNN-HGT-KG-AUDIT/kg_storage/global_graph.pt"
    GRAPH_PATH = "/Users/tanzeel.shaikh/Sources/Projects/GNN-HGT/kg_storage/global_graph_bge_small.pt"

    HIDDEN_DIM = 384 # Updated for NVIDIA NV-Embed-v2
    OUT_DIM = 128     # Contrastive projection dimension
    
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Detect hidden_dim from data if possible to ensure compatibility
    if os.path.exists(GRAPH_PATH):
        print(f"Probing {GRAPH_PATH} for dimensions...")
        temp_data = torch.load(GRAPH_PATH, map_location='cpu', weights_only=False)
        if 'entity' in temp_data.node_types:
            actual_dim = temp_data['entity'].x.shape[1]
            if actual_dim != HIDDEN_DIM:
                print(f"Note: Detected dimension {actual_dim} from graph, overriding default {HIDDEN_DIM}")
                HIDDEN_DIM = actual_dim
        del temp_data
    
    pretrain(GRAPH_PATH, HIDDEN_DIM, OUT_DIM, epochs=20, lr=1e-3, device=device, batch_size=1024)

