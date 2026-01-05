import os
import torch
import torch.nn as nn
from tqdm import tqdm
import glob
from AuditableHybridGNN_POC import AuditableHybridGNN, NVEmbedV2EmbeddingModel

# Custom Dataset Loader
class PreprocessedGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=False)

def finetune_on_noisy_data(backbone_path, data_dir, hidden_dim, epochs=5, device='cpu'):
    print(f"Using device: {device}")
    
    # 1. Load Model
    # Load a dummy graph just to get metadata
    dummy_data = torch.load(glob.glob(os.path.join(data_dir, "*.pt"))[0])
    model = AuditableHybridGNN(dummy_data['graph'].metadata(), hidden_dim).to(device)
    model.load_backbone(backbone_path, device=device, freeze=True)
    
    # 2. Setup Embedder
    embed_model = NVEmbedV2EmbeddingModel("BAAI/bge-small-en-v1.5")
    
    # 3. Optimizer & Loss
    # Tune these parameters!
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    # Pos_Weight = 20 means "Treat 1 positive error as bad as 20 negative errors"
    # This helps because we have ~150 negatives and only ~5 positives.
    #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]).to(device))
    # 3. RANKING LOSS
    # "pos_score" must be higher than "neg_score" by at least 0.5
    criterion = nn.MarginRankingLoss(margin=0.5)

    dataset = PreprocessedGraphDataset(data_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    
    accumulation_steps = 16 
    
    print("Starting Fine-tuning...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for i, sample in enumerate(pbar):
            graph = sample['graph'].to(device)
            target = sample['target'].to(device)
            query_text = sample['query_text']
            
            # Skip samples where Beam Search failed to find ANY positives (Target is all 0s)
            # Training on these confuses the model ("Everything is wrong??")
            if target.sum() == 0:
                continue
            
            # Embed Query
            q_emb = embed_model.batch_encode(query_text, instruction="query")
            q_emb = torch.from_numpy(q_emb).to(device).squeeze()
            
            # Forward (Get Logits)
            logits = model(graph.x_dict, graph.edge_index_dict, q_emb, return_logits=True)
            
            # Loss
            loss = criterion(logits, target)
            loss = loss / accumulation_steps # Normalize
            loss.backward()
            
            total_loss += loss.item()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            pbar.set_postfix({'loss': f"{total_loss / (i+1):.4f}"})
            
        # Save Checkpoint
        torch.save(model.state_dict(), f"finetuned_gnn_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    finetune_on_noisy_data(
        "models/pre-trained/pretrained_gnn_final.pth", 
        "processed_train_data", 
        384
    )