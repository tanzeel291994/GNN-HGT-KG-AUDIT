import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.data import HeteroData
import os
import json
from typing import List, Dict, Any, Tuple, Optional
from BeamSearchExtractor import BeamSearchExtractor
from QueryBiasedWalkSubgraph import NVEmbedV2EmbeddingModel
# Import the base pre-trained GNN class
from PretrainGNN import PretrainableHeteroGNN
from construct_kg import GlobalKGManager
from QueryBiasedWalkSubgraph import extract_hetero_subgraph_with_query_walk
from tqdm import tqdm
import numpy as np

def load_eval_samples(path: str, limit: int = 100):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data[:limit]

def evaluate_subgraph_extraction(
    model_checkpoint,
    dataset_path,
    hidden_dim=384,
    device='cpu',
    top_k_entities=100,
    walk_steps=5
):
    print(f"--- Subgraph Extraction Evaluation ---")
    
    # 1. Load Global Graph and Metadata
    manager = GlobalKGManager(storage_dir="kg_storage")
    global_graph, metadata = manager.load_kg()
    if global_graph is None:
        print("Error: Global KG not found in kg_storage.")
        return

    doc_to_idx = metadata['doc_to_idx']
    global_graph = global_graph.to(device)
    
    # 2. Load Pre-trained Model for Guidance
    print(f"Loading pre-trained model for neural guidance...")
    node_types = global_graph.node_types
    edge_types = global_graph.edge_types
    model = PretrainableHeteroGNN(metadata=(node_types, edge_types), hidden_dim=hidden_dim, out_dim=128)
    
    if os.path.exists(model_checkpoint):
        state_dict = torch.load(model_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: {model_checkpoint} not found. Using randomly initialized weights.")
    
    model = model.to(device)
    model.eval()

    # 3. Initialize Embedding Model
    embed_model = NVEmbedV2EmbeddingModel("BAAI/bge-small-en-v1.5")
    
    # 4. Load Eval Samples
    samples = load_eval_samples(dataset_path)
    print(f"Loaded {len(samples)} samples for evaluation.")

    # 5. Metrics Initialization
    metrics = {
        "passage_recall": [],    # % of supporting passages found per sample
        "complete_hit": [],      # 1 if ALL supporting passages are found, else 0
        "subgraph_psg_count": [], # Number of passages in the extracted subgraph
        "targets_found_in_kg": [] # Number of supporting passages that actually exist in the KG
    }
    beam_searcher = BeamSearchExtractor(
        global_graph=global_graph, 
        embedding_model=embed_model, 
        device=device
    )    
    # 6. Run Evaluation
    pbar = tqdm(samples, desc="Evaluating Subgraph Extraction")
    for sample in pbar:
        question = sample['question']
        
        # Identify Target Global Indices (Ground Truth)
        target_indices = []
        total_supporting = 0
        for p in sample['paragraphs']:
            if p.get('is_supporting', False):
                total_supporting += 1
                p_text = p['paragraph_text']
                if p_text in doc_to_idx:
                    target_indices.append(doc_to_idx[p_text])
        
        metrics["targets_found_in_kg"].append(len(target_indices) / total_supporting if total_supporting > 0 else 1.0)
        
        if not target_indices:
            continue 
        
        target_set = set(target_indices)

        # A. Embed Query
        query_emb = torch.from_numpy(embed_model.batch_encode([question], instruction="query")).to(device)

        # B. Extract Subgraph
        # We use a selective budget to see if we can keep precision high without losing recall
        # subgraph = extract_hetero_subgraph_with_query_walk(
        #     global_graph=global_graph,
        #     query_emb=query_emb,
        #     model=model,
        #     top_k_entities=50,
        #     top_k_passages=100, # Limit the subgraph size to 100 passages
        #     walk_steps=10,
        #     restart_prob=0.1    # Higher restart keeps it focused on query entities
        # )
        # RUN BEAM SEARCH
        # beam_width=50: Broad enough to catch "spouse" and "movie" nodes
        # max_hops=2: Sufficient for "Actor -> Movie -> Director" style questions
        subgraph = beam_searcher.run(
            query_text=question, 
            beam_width=50, 
            max_hops=2
        )
        # C. Check Coverage
        extracted_psg_indices = set(subgraph['passage'].n_id.tolist())
        
        hits = [1 for idx in target_indices if idx in extracted_psg_indices]
        recall = len(hits) / len(target_indices)
        all_found = 1.0 if len(hits) == len(target_indices) else 0.0
        
        metrics["passage_recall"].append(recall)
        metrics["complete_hit"].append(all_found)
        metrics["subgraph_psg_count"].append(subgraph['passage'].num_nodes)
        
        pbar.set_postfix({
            'Recall': f"{np.mean(metrics['passage_recall']):.2%}",
            'KG_Match': f"{np.mean(metrics['targets_found_in_kg']):.2%}",
            'Avg_Psg': f"{int(np.mean(metrics['subgraph_psg_count']))}"
        })

    # 7. Print Final Results
    print("\n" + "="*40)
    print("FINAL SUBGRAPH EXTRACTION RESULTS")
    print("="*40)
    print(f"Max Supporting in KG: {np.mean(metrics['targets_found_in_kg']):.2%}")
    print(f"Avg Passage Recall (within subgraph): {np.mean(metrics['passage_recall']):.2%}")
    print(f"Complete Retrieval (All Hops): {np.mean(metrics['complete_hit']):.2%}")
    print(f"Avg Subgraph Size (Passages): {np.mean(metrics['subgraph_psg_count']):.1f}")
    print("="*40)
    print("Note: 'Max Supporting in KG' is the theoretical maximum recall given your current KG.")
    print("="*40)

if __name__ == "__main__":
    MODEL_CHECKPOINT = "models/pre-trained/pretrained_gnn_final.pth"
    DATASET_PATH = "new_datasets/eval_samples.json"
    
    # We can evaluate with different budgets
    evaluate_subgraph_extraction(
        MODEL_CHECKPOINT, 
        DATASET_PATH, 
        top_k_entities=100, 
        walk_steps=5
    )