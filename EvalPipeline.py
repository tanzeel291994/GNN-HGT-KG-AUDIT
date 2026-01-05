import json
import torch
import numpy as np
from tqdm import tqdm
from construct_kg import GlobalKGManager
from BeamSearchExtractor import BeamSearchExtractor
from QueryBiasedWalkSubgraph import NVEmbedV2EmbeddingModel
from GNNReranker import GNNReranker

# --- CONFIG ---
GRAPH_PATH = "kg_storage"
FINETUNED_MODEL = "finetuned_gnn_ranking_epoch_5.pth" # Check your saved name
EVAL_DATASET = "new_datasets/eval_samples.json"
HIDDEN_DIM = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_eval_samples(path):
    with open(path, "r") as f:
        return json.load(f)[:100] # Limit to 100 for speed

def evaluate_pipeline():
    print("--- 🚀 Starting Full Pipeline Evaluation ---")
    
    # 1. Load Tools
    kg_manager = GlobalKGManager(GRAPH_PATH)
    global_graph, metadata = kg_manager.load_kg()
    doc_to_idx = metadata['doc_to_idx'] # Text -> Global ID map
    
    embed_model = NVEmbedV2EmbeddingModel("BAAI/bge-small-en-v1.5")
    
    # 2. Load Engines
    print("Initializing Beam Search (Recall Engine)...")
    # Note: We use raw embeddings for retrieval (as discussed)
    beam_searcher = BeamSearchExtractor(global_graph, embed_model, device=DEVICE)
    
    print("Initializing GNN Reranker (Precision Engine)...")
    reranker = GNNReranker(FINETUNED_MODEL, global_graph.metadata(), HIDDEN_DIM, DEVICE)
    
    # 3. Metrics
    metrics = {
        "bs_recall": [],      # Did Beam Search find the answer? (Recall)
        "gnn_hit_1": [],      # Was the #1 Reranked doc correct? (Precision)
        "gnn_recall_5": [],   # Was the answer in Top 5 Reranked?
        "gnn_recall_10": []
    }
    
    samples = load_eval_samples(EVAL_DATASET)
    
    # 4. Run Loop
    pbar = tqdm(samples, desc="Evaluating")
    for sample in pbar:
        question = sample['question']
        
        # A. Identify Ground Truth Global IDs
        target_ids = []
        for p in sample['paragraphs']:
            if p['is_supporting'] and p['paragraph_text'] in doc_to_idx:
                target_ids.append(doc_to_idx[p['paragraph_text']])
        
        if not target_ids: continue # Skip bad samples
        target_set = set(target_ids)

        # --- STEP 1: RETRIEVAL (Beam Search) ---
        # Get ~100 candidates
        subgraph = beam_searcher.run(question, beam_width=50, max_hops=2)
        
        # Check Retrieval Recall
        retrieved_ids = set(subgraph['passage'].n_id.tolist())
        hits = target_set.intersection(retrieved_ids)
        bs_recall = len(hits) / len(target_set)
        metrics["bs_recall"].append(bs_recall)
        
        if bs_recall == 0:
            # If Retrieval failed, Reranker has no chance
            metrics["gnn_hit_1"].append(0)
            metrics["gnn_recall_5"].append(0)
            metrics["gnn_recall_10"].append(0)
            continue

        # --- STEP 2: RERANKING (Fine-Tuned GNN) ---
        # Embed Query for GNN
        q_emb = torch.from_numpy(embed_model.batch_encode(question, instruction="query"))
        
        # Get Top 10 Ranked Global IDs
        ranked_ids, scores = reranker.rerank(subgraph, q_emb, top_k=10,mode="hybrid")#mode="cosine_only"
        
        # --- STEP 3: SCORING ---
        # Hit @ 1
        metrics["gnn_hit_1"].append(1 if ranked_ids[0] in target_set else 0)
        
        # Recall @ 5
        hits_5 = set(ranked_ids[:5]).intersection(target_set)
        metrics["gnn_recall_5"].append(len(hits_5) / len(target_set))
        
        # Recall @ 10
        hits_10 = set(ranked_ids[:10]).intersection(target_set)
        metrics["gnn_recall_10"].append(len(hits_10) / len(target_set))
        
        pbar.set_postfix({
            "R_Recall": f"{np.mean(metrics['bs_recall']):.2f}",
            "G_Hit@1": f"{np.mean(metrics['gnn_hit_1']):.2f}"
        })

    # 5. Final Report
    print("\n" + "="*40)
    print("FINAL PIPELINE RESULTS")
    print("="*40)
    print(f"1. Retrieval (Beam Search) Recall: {np.mean(metrics['bs_recall']):.2%}")
    print(f"   (This is the ceiling for the Reranker)")
    print("-" * 20)
    print(f"2. Reranker Hit@1 (Strict Accuracy): {np.mean(metrics['gnn_hit_1']):.2%}")
    print(f"3. Reranker Recall@5 (Top 5 Psg):    {np.mean(metrics['gnn_recall_5']):.2%}")
    print(f"4. Reranker Recall@10 (Context):     {np.mean(metrics['gnn_recall_10']):.2%}")
    print("="*40)

if __name__ == "__main__":
    evaluate_pipeline()