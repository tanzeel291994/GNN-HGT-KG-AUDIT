# train_ranking.py
import torch
import torch.nn as nn
import glob
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from AuditableHybridGNN_POC import AuditableHybridGNN, EnhancedQueryInteraction
from QueryBiasedWalkSubgraph import NVEmbedV2EmbeddingModel

class NoisyDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx): return torch.load(self.files[idx], weights_only=False)

def train_ranking(backbone_path, data_dir, hidden_dim=384, epochs=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load Stub Metadata
    dummy = torch.load(glob.glob(f"{data_dir}/*.pt")[0], weights_only=False)
    metadata = dummy['graph'].metadata()
    
    model = AuditableHybridGNN(metadata, hidden_dim).to(device)
    model.load_backbone(backbone_path, device=device, freeze=True)
    
    embed_model = NVEmbedV2EmbeddingModel("BAAI/bge-small-en-v1.5")
    
    # [NEW] Unfreeze the LAST HGT Conv layer and the Norm layers
    # This allows the GNN to adapt its "reasoning" without destroying the pre-training
    for param in model.convs[-1].parameters():
        param.requires_grad = True
    for param in model.entity_norm.parameters():
        param.requires_grad = True
    for param in model.passage_norm.parameters():
        param.requires_grad = True
        
    # Update Optimizer to include these new parameters
    # We use a lower LR for the backbone (1e-5) to avoid catastrophic forgetting
    head_params = list(map(id, model.scoring_head.parameters())) + \
                  list(map(id, model.query_interaction.parameters())) + \
                  list(map(id, model.relevance_head.parameters() if hasattr(model, 'relevance_head') else []))
    
    backbone_params = filter(lambda p: id(p) not in head_params and p.requires_grad, model.parameters())
    head_params_list = filter(lambda p: id(p) in head_params, model.parameters())

    optimizer = torch.optim.Adam([
        {'params': head_params_list, 'lr': 5e-4},   # Fast learning for Heads
        {'params': backbone_params,  'lr': 1e-5}    # Slow fine-tuning for GNN
    ])
    
    # 3. RANKING LOSS
    # "pos_score" must be higher than "neg_score" by at least 0.5
    criterion = nn.MarginRankingLoss(margin=0.1)
    
    dataset = NoisyDataset(data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    
    accumulation_steps = 16
    
    print("Starting Ranking-Based Fine-tuning...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for i, sample in enumerate(pbar):
            graph = sample['graph'].to(device)
            target = sample['target'].to(device) # 1.0 for pos, 0.0 for neg
            
            # Skip graphs with no positives (cannot rank)
            if target.sum() == 0: continue

            # Embed Query
            q_emb = embed_model.batch_encode(sample['query_text'], instruction="query")
            q_emb = torch.from_numpy(q_emb).to(device).squeeze()
            
            # Forward (Get Raw Logits)
            logits = model(graph.x_dict, graph.edge_index_dict, q_emb, return_logits=True)
            
            # --- PAIRWISE RANKING LOGIC ---
            # 1. Separate Scores
            pos_mask = (target == 1.0)
            neg_mask = (target == 0.0)
            
            pos_scores = logits[pos_mask]
            neg_scores = logits[neg_mask]
            
            if len(neg_scores) == 0: continue # All positives? Skip.
            
            # 2. Hard Negative Mining
            # Instead of comparing against ALL negatives (too easy),
            # compare against the highest scoring negatives (the confusing ones)
            # We take the top k negatives where k = num_positives
            
            # Expand pos_scores to pair with negs
            # Simple approach: Compare Mean Positive vs Mean Top-K Negatives
            # Robust approach: Compare Every Positive vs Hardest Negative
            
            hardest_neg_score = neg_scores.max()
            
            # We want every positive to be higher than the hardest negative
            # Target for MarginRankingLoss is 1 (input1 > input2) or -1 (input1 < input2)
            y = torch.ones(pos_scores.size(0)).to(device)
            
            # Broadcast hardest negative to match shape of pos_scores
            hard_neg_expanded = hardest_neg_score.expand_as(pos_scores)
            
            loss = criterion(pos_scores, hard_neg_expanded, y)
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': f"{total_loss / (i+1):.4f}"})
            
        torch.save(model.state_dict(), f"finetuned_gnn_ranking_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_ranking("models/pre-trained/pretrained_gnn_final.pth", "processed_train_data")