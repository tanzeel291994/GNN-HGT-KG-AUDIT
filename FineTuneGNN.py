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
def finetune(backbone_path, dataset_path, hidden_dim, epochs=10, lr=5e-4, device='cuda'):
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
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # 4. Training Loop
    print(f"Starting Fine-tuning on {len(samples)} samples...")
    for epoch in range(epochs):
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
            doc_texts = [p['paragraph_text'] for p in paragraphs]
            graph, _ = builder.build_graph(doc_texts)
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

if __name__ == "__main__":
    BACKBONE_PATH = "pretrained_gnn_final.pth"
    DATASET_PATH = "dataset/musique_all.json"
    HIDDEN_DIM = 384
    
    finetune(BACKBONE_PATH, DATASET_PATH, HIDDEN_DIM, epochs=5, device='cuda')

