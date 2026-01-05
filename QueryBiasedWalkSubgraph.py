import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.data import HeteroData
import os
import json
from typing import List, Dict, Any, Tuple, Optional

# Import the base pre-trained GNN class
from PretrainGNN import PretrainableHeteroGNN
from construct_kg import GlobalKGManager

# --- MINIMAL EMBEDDING WRAPPER ---
# Self-contained wrapper for embedding queries
class NVEmbedV2EmbeddingModel:
    def __init__(self, model_name: str = "nvidia/NV-Embed-v2"):
        from sentence_transformers import SentenceTransformer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()

    def batch_encode(self, texts: List[str], instruction: str = "passage", batch_size: int = 16) -> Any:
        if isinstance(texts, str): texts = [texts]
        # NV-Embed/BGE style instructions
        return self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=False,
            convert_to_numpy=True
        )

def compute_gnn_attention_weights(
    node_embs_contextualized: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    query_emb: torch.Tensor
) -> torch.Tensor:
    """
    Computes guidance weights using MULTI-HOP and GLOBAL contextualized embeddings.
    This reflects the model's full reasoning (HGT + SGFormer).
    """
    device = node_embs_contextualized.device
    row, col = edge_index
    
    # 1. Structural Affinity (Pre-trained reasoning)
    # Contextualized similarity between source and target entities.
    # Since these embeddings passed through SGFormer and multiple HGT layers,
    # this similarity encodes multi-hop and global relationships.
    src_norm = F.normalize(node_embs_contextualized[row], p=2, dim=-1)
    dst_norm = F.normalize(node_embs_contextualized[col], p=2, dim=-1)
    struct_affinity = (src_norm * dst_norm).sum(-1)
    struct_affinity = F.relu(struct_affinity)

    # 2. Factor in Query-Relation Alignment (Semantic Guidance)
    if edge_attr is not None:
        query_norm = F.normalize(query_emb, p=2, dim=-1)
        edge_attr_norm = F.normalize(edge_attr, p=2, dim=-1)
        rel_alignment = torch.mm(edge_attr_norm, query_norm.t()).squeeze()
        rel_alignment = F.relu(rel_alignment)
    else:
        rel_alignment = 1.0
        
    return struct_affinity * rel_alignment + 1e-9

def extract_hetero_subgraph_with_query_walk(
    global_graph: HeteroData,
    query_emb: torch.Tensor,
    model: Optional[PretrainableHeteroGNN] = None,
    top_k_entities: int = 50,
    top_k_passages: int = 50,
    walk_steps: int = 10,
    restart_prob: float = 0.5
) -> HeteroData:
    """
    Neural-Structural Feedback Walk using FULL pre-trained guidance.
    Improved: More selective expansion and query-aware transitions.
    """
    # 1. Contextualize Entities (The "Reasoning" Pass)
    if model is not None:
        model.eval()
        with torch.no_grad():
            node_embs = model(global_graph.x_dict, global_graph.edge_index_dict, return_h=True)
    else:
        node_embs = global_graph['entity'].x

    device = node_embs.device

    # 2. Prepare Semantic Guidance (The Restart Vector)
    # CRITICAL FIX: Use ORIGINAL embeddings for the restart vector.
    # The GNN-transformed space (node_embs) may have drifted from the query space.
    original_node_embs = global_graph['entity'].x
    q_norm = F.normalize(query_emb, p=2, dim=-1)
    orig_nodes_norm = F.normalize(original_node_embs, p=2, dim=-1)
    
    # Semantic Similarity (Base Relevance)
    semantic_sim = torch.mm(orig_nodes_norm, q_norm.t()).squeeze()
    semantic_sim = F.relu(semantic_sim)
    
    # 3. RELEVANCE (The "Gravity" toward the query)
    # We use a temperature-scaled softmax to identify starting points
    relevance = torch.exp(semantic_sim * 15.0) 
    relevance = relevance / (relevance.sum() + 1e-9)

    # 4. Prepare Structural Guidance (The Transition Matrix)
    edge_weights = None
    if ('entity', 're', 'entity') in global_graph.edge_types:
        edge_index = global_graph['entity', 're', 'entity'].edge_index
        edge_attr = global_graph['entity', 're', 'entity'].edge_attr
        
        # Guide transitions using contextualized affinity (the GNN's reasoning)
        edge_weights = compute_gnn_attention_weights(
            node_embs_contextualized=node_embs,
            edge_index=edge_index,
            edge_attr=edge_attr,
            query_emb=query_emb
        )
        
        # Factor in Semantic Bias: transitions to entities semantically 
        # related to the query are preferred.
        col = edge_index[1]
        edge_weights = edge_weights * (1.0 + semantic_sim[col])
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    # 4. Perform the walk
    entity_scores,top_entity_indices = extract_subgraph_with_query_walk(
        query_emb=query_emb,
        node_embs=node_embs,
        edge_index=edge_index,
        edge_weights=edge_weights,
        walk_steps=walk_steps,
        restart_prob=restart_prob,
        top_k=top_k_entities,
        precomputed_relevance=relevance
    )

    # 6. FILTERED EXPANSION
    # Instead of taking ALL neighbors, we filter by semantic similarity to query
    
    # A. Entity -> Sentence (Get candidates)
    e2s_idx = global_graph['entity', 'in', 'sentence'].edge_index
    ent_mask = torch.isin(e2s_idx[0], top_entity_indices)
    sentence_candidates_idx = e2s_idx[1][ent_mask]
    sentence_candidates_idx = torch.unique(sentence_candidates_idx)

    # B. Score Sentences (Crucial Step!)
    # Only keep sentences that are somewhat relevant to the query
    sent_embs = global_graph['sentence'].x[sentence_candidates_idx]
    sent_norm = F.normalize(sent_embs, p=2, dim=-1)
    # query_emb must be same dimension!
    sent_scores = torch.mm(sent_norm, q_norm.t()).squeeze()
    
    # Keep top 2 * top_k_passages sentences
    k_sent = min(len(sentence_candidates_idx), top_k_passages * 3)
    _, top_sent_local = torch.topk(sent_scores, k=k_sent)
    top_sentence_indices = sentence_candidates_idx[top_sent_local]

    # C. Sentence -> Passage
    s2p_idx = global_graph['sentence', 'in', 'passage'].edge_index
    sent_mask = torch.isin(s2p_idx[0], top_sentence_indices)
    passage_candidates = s2p_idx[1][sent_mask]

    # D. Final Ranking of Passages
    unique_passages = torch.unique(passage_candidates)
    if unique_passages.numel() > top_k_passages:
        psg_embs = global_graph['passage'].x[unique_passages]
        psg_norm = F.normalize(psg_embs, p=2, dim=-1)
        psg_scores = torch.mm(psg_norm, q_norm.t()).squeeze()
        
        _, top_psg_local = torch.topk(psg_scores, k=top_k_passages)
        final_passage_subset = unique_passages[top_psg_local]
    else:
        final_passage_subset = unique_passages

    # 8. RECONSTRUCT SUBGRAPH NODES
    # Now pull in only the sentences and entities that belong to THESE passages.
    p2s_idx = global_graph['passage', 'hv', 'sentence'].edge_index
    final_sentence_subset = torch.unique(p2s_idx[1][torch.isin(p2s_idx[0], final_passage_subset)])
    
    s2e_idx = global_graph['sentence', 'hv', 'entity'].edge_index
    final_entity_subset = torch.unique(s2e_idx[1][torch.isin(s2e_idx[0], final_sentence_subset)])
    
    # Ensure our top-scored entities are included
    final_entity_subset = torch.unique(torch.cat([final_entity_subset, top_entity_indices]))

    node_subset_dict = {
        'entity': final_entity_subset,
        'sentence': final_sentence_subset,
        'passage': final_passage_subset
    }
    subgraph = global_graph.subgraph(node_subset_dict)
    
    for node_type, subset in node_subset_dict.items():
        subgraph[node_type].n_id = subset
        
    return subgraph

def extract_subgraph_with_query_walk(
    query_emb: torch.Tensor, 
    node_embs: torch.Tensor, 
    edge_index: torch.Tensor, 
    edge_weights: Optional[torch.Tensor] = None, 
    walk_steps: int = 10, 
    restart_prob: float = 0.1, # LOWER this to allow deeper walks
    top_k: int = 50,
    precomputed_relevance: Optional[torch.Tensor] = None
):
    num_nodes = node_embs.shape[0]
    device = node_embs.device
    
    # 1. Relevance (Gravity)
    if precomputed_relevance is not None:
        relevance_prob = precomputed_relevance.squeeze()
    else:
        # Check alignment here!
        query_norm = F.normalize(query_emb, p=2, dim=-1)
        nodes_norm = F.normalize(node_embs, p=2, dim=-1)
        relevance_prob = F.relu(torch.mm(nodes_norm, query_norm.t()).squeeze())
    
    relevance_prob = relevance_prob / (relevance_prob.sum() + 1e-9)

    # 2. Transition Matrix with HUB SUPPRESSION
    row, col = edge_index
    if edge_weights is None:
        edge_weights = torch.ones(edge_index.shape[1], device=device)

    # Calculate degrees
    deg = scatter_add(edge_weights, row, dim=0, dim_size=num_nodes)
    
    # [FIX] Penalize Hubs: Divide by degree^alpha (alpha=0.5 to 1.0)
    # This prevents the walker from getting stuck in "USA" or "The"
    deg_inv = 1.0 / (deg.pow(0.8) + 1e-9) 
    
    norm_edge_weights = edge_weights * deg_inv[row]

    # 3. The Walk
    current_scores = relevance_prob.clone()
    for step in range(walk_steps):
        scattered_scores = current_scores[row] * norm_edge_weights
        new_scores = scatter_add(scattered_scores, col, dim=0, dim_size=num_nodes)
        
        # [FIX] Add a "decay" factor so deeper nodes have slightly less weight than start nodes
        current_scores = (1 - restart_prob) * new_scores + restart_prob * relevance_prob

    _, top_indices = torch.topk(current_scores, k=min(top_k, num_nodes))
    return _,top_indices

def extract_subgraph_with_query_walk_(
    query_emb: torch.Tensor, 
    node_embs: torch.Tensor, 
    edge_index: torch.Tensor, 
    edge_weights: Optional[torch.Tensor] = None, 
    walk_steps: int = 10, 
    restart_prob: float = 0.2, 
    top_k: int = 50,
    precomputed_relevance: Optional[torch.Tensor] = None
):
    """
    Performs a query-biased random walk.
    """
    num_nodes = node_embs.shape[0]
    device = node_embs.device
    
    # 1. THE RELEVANCE VECTOR (The "Gravity")
    if precomputed_relevance is not None:
        relevance_prob = precomputed_relevance.squeeze()
    else:
        query_norm = F.normalize(query_emb, p=2, dim=-1)
        nodes_norm = F.normalize(node_embs, p=2, dim=-1)
        similarity_scores = torch.mm(nodes_norm, query_norm.t()).squeeze() 
        relevance_prob = F.relu(similarity_scores)
    
    relevance_prob = relevance_prob / (relevance_prob.sum() + 1e-9)

    # 2. THE TRANSITION MATRIX
    row, col = edge_index
    if edge_weights is None:
        # Default to uniform if no GNN weights provided
        edge_weights = torch.ones(edge_index.shape[1], device=device)

    # Row-normalize to get transition probabilities
    deg = scatter_add(edge_weights, row, dim=0, dim_size=num_nodes)
    deg_inv = 1.0 / (deg + 1e-9)
    norm_edge_weights = edge_weights * deg_inv[row]

    # 3. THE WALK (Power Iteration)
    current_scores = relevance_prob.clone()
    for step in range(walk_steps):
        scattered_scores = current_scores[row] * norm_edge_weights
        new_scores = scatter_add(scattered_scores, col, dim=0, dim_size=num_nodes)
        # Random Walk with Restart (RWR)
        current_scores = (1 - restart_prob) * new_scores + restart_prob * relevance_prob

    # 4. EXTRACTION
    _, top_indices = torch.topk(current_scores, k=min(top_k, num_nodes))
    return top_indices, current_scores


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Setup paths
    model_checkpoint = "models/pre-trained/pretrained_gnn_final.pth"
    kg_manager = GlobalKGManager(storage_dir="kg_storage")
    
    # 2. Load Global Graph
    print("Loading Global KG...")
    global_graph, metadata = kg_manager.load_kg()
    if global_graph is None:
        print("Error: Global graph not found.")
        exit()
    
    # 3. Load PRE-TRAINED Model (Not the POC/Fine-tuned one)
    print("Loading Pre-trained Model...")
    node_types = global_graph.node_types
    edge_types = global_graph.edge_types
    # hidden_dim=384, out_dim=128 as per PretrainGNN.py
    model = PretrainableHeteroGNN(metadata=(node_types, edge_types), hidden_dim=384, out_dim=128)
    
    if os.path.exists(model_checkpoint):
        state_dict = torch.load(model_checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
        print("Pre-trained weights loaded.")
    else:
        print(f"Warning: {model_checkpoint} not found. Using randomly initialized weights.")
    
    model.eval()
    
    # 4. Init Embedding Model (to embed queries)
    embed_model = NVEmbedV2EmbeddingModel("BAAI/bge-small-en-v1.5")
    
    # 5. TEST: Run a query
    test_query = "Who is the spouse of the Green performer?"
    print(f"\nQuery: {test_query}")
    
    # A. Embed Query
    query_emb = torch.from_numpy(embed_model.batch_encode([test_query], instruction="query"))
    
    # B. Perform Query-Biased Walk Subgraph Extraction
    print("Performing Neural-Structural Feedback Walk...")
    subgraph = extract_hetero_subgraph_with_query_walk(
        global_graph=global_graph,
        query_emb=query_emb,
        model=model, # Now using the pre-trained model for learned guidance
        top_k_entities=50,
        walk_steps=10,
        restart_prob=0.2
    )
    
    print(f"Extracted Subgraph: {subgraph['entity'].num_nodes} entities, "
          f"{subgraph['sentence'].num_nodes} sentences, "
          f"{subgraph['passage'].num_nodes} passages.")
    
    # 6. Inspect Results
    print("\nExtraction Complete.")
    print(f"Top 10 Global Entity IDs: {subgraph['entity'].n_id[:10].tolist()}")
    print(f"Top 10 Global Passage IDs: {subgraph['passage'].n_id[:10].tolist()}")

