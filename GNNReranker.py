import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
# Import your model class definition
from AuditableHybridGNN_POC import AuditableHybridGNN, EnhancedQueryInteraction

class GNNReranker:
    def __init__(self, model_path, global_metadata, hidden_dim=384, device='cuda'):
        self.device = device
        print(f"Loading Fine-Tuned GNN from {model_path}...")
        
        # Initialize Architecture
        self.model = AuditableHybridGNN(global_metadata, hidden_dim).to(device)
        
        # Load Weights
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("✅ Reranker Loaded.")

    def rerank(self, subgraph: HeteroData, query_emb: torch.Tensor, top_k=10, mode="hybrid"):
        """
        mode: 'hybrid' (GNN + Cosine), 'gnn_only', 'cosine_only'
        """
        subgraph = subgraph.to(self.device)
        if query_emb.dim() == 1: query_emb = query_emb.unsqueeze(0)
        query_emb = query_emb.to(self.device)

        with torch.no_grad():
            # 1. Get GNN Score (Pure Structure)
            gnn_logits = self.model(subgraph.x_dict, subgraph.edge_index_dict, query_emb, return_logits=True)
            gnn_probs = torch.sigmoid(gnn_logits)
            
            # 2. Get BGE Score (Pure Semantic)
            p_norm = F.normalize(subgraph.x_dict['passage'], p=2, dim=-1)
            q_norm = F.normalize(query_emb, p=2, dim=-1)
            cosine_scores = torch.matmul(p_norm, q_norm.t()).squeeze()
            
            # 3. ENSEMBLE: 80% BGE + 20% GNN
            # This guarantees BGE accuracy but allows GNN to bump up structurally relevant nodes
            final_scores = (0.80 * cosine_scores) + (0.20 * gnn_probs)
        
        # Sort and Select
        k = min(top_k, final_scores.size(0))
        top_scores, local_indices = torch.topk(final_scores, k=k)
        global_ids = subgraph['passage'].n_id[local_indices]
        
        return global_ids.tolist(), top_scores.tolist()

    def rerank_(self, subgraph: HeteroData, query_emb: torch.Tensor, top_k=10):
        """
        Input: 
            subgraph: The ~150 node graph from Beam Search
            query_emb: The [1, Dim] embedding of the question
        Output:
            List of Global Passage IDs (sorted by relevance)
        """
        subgraph = subgraph.to(self.device)
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        query_emb = query_emb.to(self.device)

        with torch.no_grad():
            # 1. Forward Pass (Get Probabilities)
            # We use the 'combined_scores' (Late Fusion) or just raw logits depending on your preference
            # Default forward() returns combined_scores [0..1]
            scores = self.model(subgraph.x_dict, subgraph.edge_index_dict, query_emb)
            print(f"DEBUG Stats:")
            print(f"  Max GNN Score: {scores.max().item():.4f}")
            print(f"  Min GNN Score: {scores.min().item():.4f}")
            print(f"  Mean GNN Score: {scores.mean().item():.4f}")
        # 2. Sort
        # These indices are LOCAL to the subgraph (0 to 150)
        k = min(top_k, scores.size(0))
        top_scores, local_indices = torch.topk(scores, k=k)
        
        # 3. Map to Global IDs
        # subgraph['passage'].n_id contains the mapping [Local -> Global]
        global_ids = subgraph['passage'].n_id[local_indices]
        
        return global_ids.tolist(), top_scores.tolist()

    def rerank_(self, subgraph: HeteroData, query_emb: torch.Tensor, top_k=10, mode="hybrid"):
        """
        mode: 'hybrid' (GNN + Cosine), 'gnn_only', 'cosine_only'
        """
        subgraph = subgraph.to(self.device)
        if query_emb.dim() == 1: query_emb = query_emb.unsqueeze(0)
        query_emb = query_emb.to(self.device)

        with torch.no_grad():
            # Get RAW GNN Logits
            gnn_logits = self.model(subgraph.x_dict, subgraph.edge_index_dict, query_emb, return_logits=True)
            gnn_probs = torch.sigmoid(gnn_logits)
            
            # Recalculate Cosine Similarity (Initial Scores) manually for mixing
            p_norm = F.normalize(subgraph.x_dict['passage'], p=2, dim=-1)
            q_norm = F.normalize(query_emb, p=2, dim=-1)
            cosine_scores = torch.matmul(p_norm, q_norm.t()).squeeze()
            
            if mode == "cosine_only":
                final_scores = cosine_scores
            elif mode == "gnn_only":
                final_scores = gnn_probs
            else: # Hybrid
                # HEURISTIC: Weight Cosine higher (0.7) because BGE is very strong
                final_scores = (0.7 * cosine_scores) + (0.3 * gnn_probs)
        
        # Sort and Select
        k = min(top_k, final_scores.size(0))
        top_scores, local_indices = torch.topk(final_scores, k=k)
        global_ids = subgraph['passage'].n_id[local_indices]
        
        return global_ids.tolist(), top_scores.tolist()