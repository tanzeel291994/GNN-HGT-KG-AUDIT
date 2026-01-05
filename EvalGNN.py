import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from AuditableHybridGNN_POC import AuditableHybridGNN, POCGraphBuilder, NVEmbedV2EmbeddingModel
from tqdm import tqdm
import numpy as np

from FineTuneGNN import get_or_build_graph
from construct_kg import GlobalKGManager

# --- 1. DATA LOADING FOR EVALUATION ---
def load_musique_eval_samples(path: str, limit: int = 200):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data[:limit]

# --- 2. EVALUATION SCRIPT ---
def evaluate(model_path, dataset_path, hidden_dim, device='cuda', k_list=[2, 3, 5, 10]):
    print("Initializing Model and loading fine-tuned weights...")
    
    # Load metadata for initialization
    temp_graph = torch.load("kg_storage/global_graph_bge_small.pt", map_location='cpu', weights_only=False)
    metadata = temp_graph.metadata()
    del temp_graph
    
    model = AuditableHybridGNN(metadata, hidden_dim).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading fine-tuned model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    else:
        print(f"Warning: {model_path} not found. Running zero-shot evaluation.")
    
    model.eval()
    
    # Setup Tools
    embed_model = NVEmbedV2EmbeddingModel("BAAI/bge-small-en-v1.5")
    azure_config = {
        "api_key": "db8dac9cea3148d48c348ed46e9bfb2d",
        "endpoint": "https://bodeu-des-csv02.openai.azure.com/",
        "api_version": "2024-12-01-preview", 
        "deployment_name": "gpt-4o-mini" 
    }    
    builder = POCGraphBuilder(embed_model,azure_config, cache_path="musique_ie_cache.json")
    
    samples = load_musique_eval_samples(dataset_path)
    print(f"Evaluating on {len(samples)} samples from {dataset_path}...")
    
    recalls = {k: [] for k in k_list}
    mrr = []
    all_hops_found = {k: [] for k in k_list}
    manager = GlobalKGManager()
    global_graph, metadata = manager.load_kg()
    # HYDRATE THE BUILDER: This is the critical fix
    # Ensure the builder knows the entity mapping used in the global graph
    builder.entity_to_idx = metadata['entity_to_idx']
    # Reconstruct unique_entities list from the mapping to allow index-based printing
    builder.unique_entities = [None] * len(metadata['entity_to_idx'])
    for ent, idx in metadata['entity_to_idx'].items():
        builder.unique_entities[idx] = ent

    doc_to_idx = metadata['doc_to_idx'] # Map: paragraph_text -> global_index

    with torch.no_grad():
        pbar = tqdm(samples[:1], desc="Evaluating")
        for sample in pbar:
            question = sample['question']
            target_global_indices = []
            for p in sample['paragraphs']:
                if p.get('is_supporting', False):
                    p_text = p['paragraph_text']
                    if p_text in doc_to_idx:
                        target_global_indices.append(doc_to_idx[p_text])
            
            # Convert to set for O(1) lookup
            target_indices = set(target_global_indices)
            

            # --- NEW SUBGRAPH EXTRACTION LOGIC ---
            # Step A: Get Query Embedding
            query_emb = torch.from_numpy(embed_model.batch_encode([question], instruction="query")).to(device)
            
            # Step B: Identify Anchor Entities (Physical + Semantic)
            anchors = builder.extract_query_anchors(question)
            print(f"Anchors: {anchors}")
            # Step C: Extract Subgraph from Global Graph (Algorithm 5 Hybrid)
            graph = builder.build_attention_guided_subgraph(
                global_graph, 
                anchors, 
                query_emb=query_emb
                #k_semantic=50
            )
            
            if graph is None:
                # Handle cases where no anchors are found
                print(f"No anchors found for question: {question}")
                continue
            
            graph = graph.to(device)
            
            # Step D: Forward Pass
            #scores = model(graph.x_dict, graph.edge_index_dict, query_emb)


            # 1. Build Subgraph from the sample
            #doc_texts = [p['paragraph_text'] for p in paragraphs]
            #graph, _ = builder.build_graph(doc_texts)
            #graph = graph.to(device)
            # 1. Get or Build Graph (This will be FAST after the 1st epoch)
            #graph = get_or_build_graph(sample, builder, storage_dir="graph_storage")
            #graph = graph.to(device)
            # 2. Get Query Embedding
            #query_emb = torch.from_numpy(embed_model.batch_encode([question], instruction="query")).to(device)
            
            # 3. Forward Pass
            scores = model(graph.x_dict, graph.edge_index_dict, query_emb)
            if scores.dim() == 0: scores = scores.unsqueeze(0)
            
            # 4. Rank indices
            # Applying sigmoid because we used BCEWithLogitsLoss
            probs = torch.sigmoid(scores)
            sorted_local_indices = torch.argsort(probs, descending=True).tolist()
            
            # CRITICAL FIX: Map local subgraph indices back to global IDs
            # graph['passage'].n_id contains the mapping [local_idx] -> global_idx
            subgraph_global_ids = graph['passage'].n_id.tolist()
            sorted_global_indices = [subgraph_global_ids[idx] for idx in sorted_local_indices]
            
            # 5. Calculate Metrics
            # MRR Calculation (for the first supporting paragraph found)
            rank = 999
            for i, idx in enumerate(sorted_global_indices):
                if idx in target_indices:
                    rank = i + 1
                    break
            mrr.append(1.0 / rank if rank != 999 else 0.0)
            
            for k in k_list:
                top_k_indices = sorted_global_indices[:k]
                
                # Recall@K: Did we find ANY of the supporting paragraphs?
                hits = [1 for idx in target_indices if idx in top_k_indices]
                recalls[k].append(len(hits) / len(target_indices))
                
                # All-Hops-Found@K: Did we find ALL supporting paragraphs in Top-K?
                all_hops_found[k].append(1.0 if len(hits) == len(target_indices) else 0.0)
            
            pbar.set_postfix({'MRR': f"{np.mean(mrr):.4f}", 'R@5': f"{np.mean(recalls[5]):.4f}"})
            
    print("\n--- FINAL EVALUATION RESULTS ---")
    print(f"Dataset: {dataset_path}")
    print(f"MRR: {np.mean(mrr):.4f}")
    for k in k_list:
        print(f"Recall@{k}: {np.mean(recalls[k]):.2%}")
        print(f"Complete-Retrieval@{k}: {np.mean(all_hops_found[k]):.2%}")
    print("---------------------------------")
    # At the very end of evaluate() in EvalGNN.py
    print("Saving updated IE cache...")
    builder._save_cache()
    print("Evaluation completed.")

if __name__ == "__main__":
    MODEL_PATH = "models/fine-tuned/finetuned_gnn_epoch_5.pth" # Update this to your best epoch
    DATASET_PATH = "new_datasets/eval_samples.json"
    HIDDEN_DIM = 384
    
    evaluate(MODEL_PATH, DATASET_PATH, HIDDEN_DIM, device='cpu')

