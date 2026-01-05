import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

class BeamSearchExtractor:
    def __init__(self, global_graph: HeteroData, embedding_model, gnn_model=None, device='cuda'):
        self.graph = global_graph
        self.embed_model = embedding_model
        self.device = device
        
        # --- CRITICAL FIX: ALWAYS USE RAW EMBEDDINGS FOR SCORING ---
        # We must align with the Query Encoder (BGE-Small)
        print("✅ Using RAW Entity Embeddings for Semantic Scoring (Fixing Space Mismatch).")
        self.raw_entity_embs = F.normalize(self.graph['entity'].x, p=2, dim=-1).to(device)
        self.sentence_embs = F.normalize(self.graph['sentence'].x, p=2, dim=-1).to(device)

        # (Optional) We can store GNN embeddings for future use (e.g., reranking), 
        # but we won't use them for the primary beam score.
        self.gnn_entity_embs = None
        if gnn_model is not None:
             gnn_model.eval()
             with torch.no_grad():
                 self.gnn_entity_embs = gnn_model(
                    self.graph.x_dict, 
                    self.graph.edge_index_dict, 
                    return_h=True
                 )
        
        # Cache edge indices
        self.e2e_idx = self.graph['entity', 're', 'entity'].edge_index.to(device)
        self.e2s_idx = self.graph['entity', 'in', 'sentence'].edge_index.to(device)
        self.s2p_idx = self.graph['sentence', 'in', 'passage'].edge_index.to(device)

    def run(self, query_text: str, beam_width: int = 50, max_hops: int = 2) -> HeteroData:
        # 1. Embed Query
        query_emb = self.embed_model.batch_encode(query_text, instruction="query")
        query_emb = torch.from_numpy(query_emb).to(self.device)
        q_norm = F.normalize(query_emb, p=2, dim=-1)

        # --- HOP 0: INITIAL RETRIEVAL ---
        # USE RAW EMBEDDINGS (Matches Query Space)
        similarity = torch.mm(self.raw_entity_embs, q_norm.t()).squeeze()
        top_scores, top_indices = torch.topk(similarity, k=beam_width)
        
        current_beam_indices = top_indices
        accumulated_nodes = set(top_indices.tolist())

        # --- HOP N: EXPAND BEAM ---
        for hop in range(max_hops):
            mask = torch.isin(self.e2e_idx[0], current_beam_indices)
            neighbors = self.e2e_idx[1][mask]
            neighbors = torch.unique(neighbors)
            
            if neighbors.numel() == 0: break

            # USE RAW EMBEDDINGS (Matches Query Space)
            neighbor_sim = torch.mm(self.raw_entity_embs[neighbors], q_norm.t()).squeeze()
            scores = neighbor_sim 
            
            k = min(beam_width, scores.numel())
            _, best_local_indices = torch.topk(scores, k=k)
            best_global_indices = neighbors[best_local_indices]
            
            current_beam_indices = best_global_indices
            accumulated_nodes.update(best_global_indices.tolist())

        # ... (Context Retrieval remains the same) ...
        final_entity_indices = torch.tensor(list(accumulated_nodes), device=self.device)
        sent_mask = torch.isin(self.e2s_idx[0], final_entity_indices)
        candidate_sentences = torch.unique(self.e2s_idx[1][sent_mask])
        
        if candidate_sentences.numel() > 0:
            sent_sim = torch.mm(self.sentence_embs[candidate_sentences], q_norm.t()).squeeze()
            k_sent = min(200, sent_sim.numel())
            _, top_sent_local = torch.topk(sent_sim, k=k_sent)
            final_sentence_indices = candidate_sentences[top_sent_local]
        else:
            final_sentence_indices = torch.empty(0, dtype=torch.long, device=self.device)

        psg_mask = torch.isin(self.s2p_idx[0], final_sentence_indices)
        final_passage_indices = torch.unique(self.s2p_idx[1][psg_mask])

        node_dict = {
            'entity': final_entity_indices,
            'sentence': final_sentence_indices,
            'passage': final_passage_indices
        }
        
        subgraph = self.graph.subgraph(node_dict)
        # Re-attach IDs
        subgraph['entity'].n_id = final_entity_indices
        subgraph['sentence'].n_id = final_sentence_indices
        subgraph['passage'].n_id = final_passage_indices
        
        return subgraph