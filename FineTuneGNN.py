import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from AuditableHybridGNN_POC import AuditableHybridGNN, POCGraphBuilder, NVEmbedV2EmbeddingModel
from tqdm import tqdm
import numpy as np

# --- 1. DATA LOADING FOR SUBGRAPH FINE-TUNING ---
def load_musique_finetune_samples(path: str, limit: int = 500):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data[:limit]

# --- 2. FINE-TUNING SCRIPT ---
def finetune(backbone_path, dataset_path, hidden_dim, epochs=10,lr_head=5e-4, lr_backbone=5e-5, device='cpu'):
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    print(f"Using device: {device}")
    # 1. Initialize Models
    # Note: We need a temporary graph to get metadata for initialization
    # (Just using a placeholder metadata or loading from existing graph)
    print("Initializing Model and loading pre-trained backbone...")
    
    # We'll load the full graph just to get the metadata structure
    temp_graph = torch.load("kg_storage/global_graph_bge_small.pt", map_location='cpu', weights_only=False)
    metadata = temp_graph.metadata()
    del temp_graph
    
    model = AuditableHybridGNN(metadata, hidden_dim).to(device)
    
    # Use the unified backbone loader
    model.load_backbone(backbone_path, device=device, freeze=True)
    
    # 3. Setup Tools
    embed_model = NVEmbedV2EmbeddingModel("BAAI/bge-small-en-v1.5")
    builder = POCGraphBuilder(embed_model, {"api_key": "", "endpoint": "", "api_version": "", "deployment_name": ""}, cache_path="musique_ie_cache.json")
    
    samples = load_musique_finetune_samples(dataset_path)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    pos_weight = torch.tensor([10.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 3. Setup Differential Learning Rates
    # Separate parameters for task-specific heads vs pretrained backbone
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if any(x in name for x in ["query_interaction", "scoring_head", "passage_norm"]):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': head_params, 'lr': lr_head},
        {'params': backbone_params, 'lr': lr_backbone}
    ])

    # 4. Training Loop
    print(f"Starting Fine-tuning on {len(samples)} samples...")
    for epoch in range(epochs):
        # --- PARTIAL UNFREEZING LOGIC ---
        if epoch == 2: # e.g., Unfreeze the last HGT layer after 2 epochs
            print(">>> Unfreezing the last HGT layer for fine-tuning...")
            for param in model.convs[-1].parameters():
                param.requires_grad = True
        
        model.train()
        epoch_loss = 0
        correct_retrievals = 0
        
        pbar = tqdm(enumerate(samples), desc=f"Epoch {epoch+1}", total=len(samples))
        for i, sample in pbar:
            question = sample['question']
            paragraphs = sample['paragraphs']
            
            # Find the indices of supporting paragraphs
            target_indices = [idx for idx, p in enumerate(paragraphs) if p['is_supporting']]
            if not target_indices: continue
            
            # 1. Build Subgraph (The "Minigraph") from the sample's passages
            #doc_texts = [p['paragraph_text'] for p in paragraphs]
            #graph, _ = builder.build_graph(doc_texts)
            #graph = graph.to(device)
            # 1. Get or Build Graph (This will be FAST after the 1st epoch)
            graph = get_or_build_graph(sample, builder, storage_dir="graph_storage")
            graph = graph.to(device)
            # 2. Get Query Embedding
            query_emb = torch.from_numpy(embed_model.batch_encode([question], instruction="query")).to(device)
            
            # 3. Forward Pass
            optimizer.zero_grad()
            # --- Updated Training Step for Multi-hop ---

            # 1. Create a multi-hot target vector
            target = torch.zeros(len(paragraphs), device=device)
            for idx in target_indices:
                target[idx] = 1.0

            # 2. Forward pass (returns raw scores/logits)
            scores = model(graph.x_dict, graph.edge_index_dict, query_emb)

            # 3. Calculate Multi-label Loss
            loss = criterion(scores, target)

            loss.backward()
            optimizer.step()

            # --- UPDATED MULTI-HOP METRICS ---
            epoch_loss += loss.item()
            
            # Sort scores to get rankings
            sorted_indices = torch.argsort(scores, descending=True).tolist()
            
            # 1. Hit@1 (Strict: was the top choice correct?)
            if sorted_indices[0] in target_indices:
                correct_retrievals += 1
            
            # 2. Recall@5 (Did we find supporting paragraphs in Top 5?)
            top_5 = sorted_indices[:5]
            hits_in_top_5 = [idx for idx in target_indices if idx in top_5]
            recall_5 = len(hits_in_top_5) / len(target_indices)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'hit@1': f"{correct_retrievals/(i+1):.2%}",
                'r@5': f"{recall_5:.2%}"
            })            
        avg_loss = epoch_loss / len(samples)
        accuracy = correct_retrievals / len(samples)
        print(f"Epoch {epoch+1:02d} | Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")
        
        # Save check-pointed fine-tuned model
        torch.save(model.state_dict(), f"finetuned_gnn_epoch_{epoch+1}.pth")

    print("Fine-tuning completed.")

import os
import torch
from hashlib import md5

def get_sample_hash(sample):
    # Create a unique ID based on the question and the set of paragraph texts
    content = sample['question'] + "".join([p['paragraph_text'] for p in sample['paragraphs']])
    return md5(content.encode()).hexdigest()

def get_or_build_graph(sample, builder, storage_dir="graph_storage"):
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
        
    sample_hash = get_sample_hash(sample)
    graph_path = os.path.join(storage_dir, f"{sample_hash}.pt")
    
    if os.path.exists(graph_path):
        # Load from disk
        return torch.load(graph_path, weights_only=False)
    else:
        # Build from scratch
        doc_texts = [p['paragraph_text'] for p in sample['paragraphs']]
        graph, _ = builder.build_graph(doc_texts)
        # Save to disk for next time
        torch.save(graph, graph_path)
        return graph

if __name__ == "__main__":
    BACKBONE_PATH = "pretrained_gnn_final.pth"
    DATASET_PATH = "new_datasets/train_samples.json"
    HIDDEN_DIM = 384
    
    finetune(BACKBONE_PATH, DATASET_PATH, HIDDEN_DIM, epochs=5)

