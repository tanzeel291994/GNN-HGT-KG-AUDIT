# generate_noisy_training_data.py
import os
import torch
import json
from tqdm import tqdm
from construct_kg import GlobalKGManager
from BeamSearchExtractor import BeamSearchExtractor
from QueryBiasedWalkSubgraph import NVEmbedV2EmbeddingModel

def generate_data():
    # 1. Setup
    OUTPUT_DIR = "processed_noisy_train"
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    kg_manager = GlobalKGManager("kg_storage")
    global_graph, metadata = kg_manager.load_kg()
    doc_to_idx = metadata['doc_to_idx']
    
    embed_model = NVEmbedV2EmbeddingModel("BAAI/bge-small-en-v1.5")
    extractor = BeamSearchExtractor(global_graph, embed_model, device='cpu')
    
    # 2. Load Train Samples
    with open("new_datasets/train_samples.json", "r") as f:
        samples = json.load(f)
        
    print(f"Generating noisy subgraphs for {len(samples)} samples...")
    
    success_count = 0
    for i, sample in tqdm(enumerate(samples), total=len(samples)):
        # A. Run Beam Search (Get ~100-150 nodes)
        # We use the same settings as Inference!
        try:
            subgraph = extractor.run(sample['question'], beam_width=50, max_hops=2)
        except Exception as e:
            continue

        # B. Create Target Labels
        # We need to find which of these 150 nodes are actually the answer
        supporting_ids = []
        for p in sample['paragraphs']:
            if p['is_supporting'] and p['paragraph_text'] in doc_to_idx:
                supporting_ids.append(doc_to_idx[p['paragraph_text']])
        
        # Map Global IDs -> Subgraph Local IDs
        # subgraph['passage'].n_id is [Local_ID -> Global_ID]
        subgraph_global_ids = subgraph['passage'].n_id.tolist()
        
        target = torch.zeros(subgraph['passage'].num_nodes)
        found = False
        
        for global_id in supporting_ids:
            if global_id in subgraph_global_ids:
                local_idx = subgraph_global_ids.index(global_id)
                target[local_idx] = 1.0
                found = True
        
        # Only save if we actually found the answer in the beam search
        # (We can't learn to find an answer that isn't there)
        if found:
            torch.save({
                'graph': subgraph.to('cpu'), 
                'query': sample['question'],
                'target': target
            }, f"{OUTPUT_DIR}/sample_{i}.pt")
            success_count += 1

    print(f"✅ Generated {success_count} noisy training graphs.")

if __name__ == "__main__":
    generate_data()