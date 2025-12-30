import os
import json
import torch
import numpy as np
from tqdm import tqdm
from AuditableHybridGNN_POC import POCGraphBuilder, NVEmbedV2EmbeddingModel

class GlobalKGManager:
    def __init__(self, storage_dir="kg_storage"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.graph_path = os.path.join(storage_dir, "global_graph_bge_small.pt")
        self.meta_path = os.path.join(storage_dir, "metadata_bge_small.json")

    def save_kg(self, data, entity_map, doc_map):
        # Save the PyG HeteroData object
        torch.save(data, self.graph_path)
        
        # Save text-to-index mappings so we can find nodes later
        metadata = {
            "entity_to_idx": entity_map,
            "doc_to_idx": doc_map
        }
        with open(self.meta_path, 'w') as f:
            json.dump(metadata, f)
        print(f"KG stored successfully in {self.storage_dir}")

    def load_kg(self):
        if not os.path.exists(self.graph_path):
            return None, None
        data = torch.load(self.graph_path)
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
        return data, meta

# --- Execution Script ---
if __name__ == "__main__":
    # 1. Config
    azure_config = {
        "api_key": "db8dac9cea3148d48c348ed46e9bfb2d",
        "endpoint": "https://bodeu-des-csv02.openai.azure.com/",
        "api_version": "2024-12-01-preview", 
        "deployment_name": "gpt-4o-mini" 
    }    
    
    corpus_path = "dataset/musique_corpus.json"
    
    # 2. Init Tools
    # Choice 1: Smaller, faster model (384 dim) - Recommended for avoiding OOM
    model_name = "BAAI/bge-small-en-v1.5"
    # Choice 2: Medium model (768 dim)
    # model_name = "BAAI/bge-base-en-v1.5"
    
    embed_model = NVEmbedV2EmbeddingModel(model_name)
    builder = POCGraphBuilder(embed_model, azure_config, cache_path="musique_ie_cache.json")
    manager = GlobalKGManager()

    # 3. Load All Data
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    
    # Format docs as "Title\nText"
    all_docs = [f"{d['text']}" for d in corpus]
    
    print(f"Building Global KG for {len(all_docs)} documents...")
    
    # 4. Build Graph (using your existing POCGraphBuilder)
    # Note: You might need to adjust build_graph to return the ent_to_idx map
    global_graph, unique_entities = builder.build_graph(all_docs)
    
    # 5. Persistent Storage
    ent_to_idx = {ent: i for i, ent in enumerate(unique_entities)}
    doc_to_idx = {doc: i for i, doc in enumerate(all_docs)}
    
    manager.save_kg(global_graph, ent_to_idx, doc_to_idx)