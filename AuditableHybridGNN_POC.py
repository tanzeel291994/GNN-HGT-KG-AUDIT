import json
import os
import re
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from hashlib import md5
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from transformers import AutoModel
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.utils import scatter
import torch.nn.functional as F
import sys
from PretrainGNN import PretrainableHeteroGNN
from unittest.mock import MagicMock
from torch_geometric.nn.models import SGFormer
# 1. Add the QSGNN path to sys.path
#qsgnn_path = os.path.abspath(os.path.join(os.getcwd(), "../QSGNN/src"))
#if qsgnn_path not in sys.path:
#    sys.path.insert(0, qsgnn_path)

# 2. MOCK the modules that require vllm or cause multiprocessing issues on Mac
# This prevents the root __init__.py from crashing when you import OpenIE.
#sys.modules["vllm"] = MagicMock()
#sys.modules["qsgnn_rag.QSGNNRAG"] = MagicMock()
#os.environ["OPENAI_API_KEY"] = ""
# 3. Now you can use the standard import! 
# It will no longer crash or complain about missing parent packages.
#from qsgnn_rag.information_extraction.openie_openai import OpenIE
#from qsgnn_rag.llm.openai_gpt import CacheOpenAI
from openai import AzureOpenAI

# --- 1. AZURE OPENIE IMPLEMENTATION ---
class SimpleAzureOpenIE:
    def __init__(self, api_key: str, endpoint: str, api_version: str, deployment_name: str):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        self.deployment_name = deployment_name

    def extract_info(self, text: str):
        """Simple NER and Triple extraction using a single Azure OpenAI call with retry logic."""
        import time
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 10]
        
        prompt = f"""
        Extract Named Entities (people, places, concepts) and semantic Triples from the text.
        Return ONLY a JSON object.
        
        Text: {text}
        
        Format:
        {{
            "entities": ["Entity 1", "Entity 2"],
            "triples": [["Subject", "Relation", "Object"]]
        }}
        """
        
        max_retries = 3
        retry_delay = 5 # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that extracts knowledge graph data in JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                if content:
                    res = json.loads(content)
                    entities = res.get("entities", [])
                    triples = res.get("triples", [])
                    return sentences, entities, triples, [
                        [ent for ent in entities if ent.lower() in sent.lower()]
                        for sent in sentences
                    ]
                else:
                    finish_reason = response.choices[0].finish_reason
                    print(f"Azure OpenAI Warning: Received empty content. Finish reason: {finish_reason}")
                    if finish_reason == "content_filter":
                        break # Don't retry for content filter
            except Exception as e:
                print(f"Azure OpenAI Error (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Sleeping for {retry_delay}s before retry...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Skipping.")

        return sentences, [], [], [[] for _ in sentences]

# --- 2. EMBEDDING MODEL (NV-Embed-v2) ---
# 2. Fix the Embedding Model class
# class NVEmbedV2EmbeddingModel:
#     def __init__(self, model_name: str = "nvidia/NV-Embed-v2"):
#         # For the POC, we'll use SentenceTransformer which provides the .encode() method
#         from sentence_transformers import SentenceTransformer
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Loading {model_name} on {self.device}...")
#         self.model = SentenceTransformer(model_name, device=self.device)
        
#     def batch_encode(self, texts: List[str], instruction: str = "", batch_size: int = 4) -> np.ndarray:
#         if isinstance(texts, str): texts = [texts]
#         # SentenceTransformer handles the encoding directly
#         results = self.model.encode(
#             texts, 
#             batch_size=batch_size,
#             show_progress_bar=False,
#             convert_to_numpy=True
#         )
#         # L2 Normalization
#         norms = np.linalg.norm(results, axis=1, keepdims=True)
#         return results / np.maximum(norms, 1e-12)

class NVEmbedV2EmbeddingModel:
    def __init__(self, model_name: str = "nvidia/NV-Embed-v2"):
        from transformers import AutoModel, AutoTokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")
        
        # Check if it's an NV-Embed model or a standard Sentence Transformer / BGE model
        self.is_nv_embed = "NV-Embed" in model_name
        
        if self.is_nv_embed:
            self.model = AutoModel.from_pretrained(
                model_name, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            ).to(self.device)
        else:
            # Fallback to SentenceTransformer for smaller models
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=self.device)
        
        self.model.eval()

    def batch_encode(self, texts: List[str], instruction: str = "passage", batch_size: int = 4) -> np.ndarray:
        if isinstance(texts, str): texts = [texts]
        
        if self.is_nv_embed:
            all_embeddings = []
            from tqdm import tqdm
            pbar = tqdm(total=len(texts), desc=f"Embedding {instruction}s", unit="text")
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                with torch.no_grad():
                    embeddings = self.model.encode(
                        batch, 
                        instruction=instruction,
                        batch_size=batch_size
                    )
                    
                    if torch.is_tensor(embeddings):
                        embeddings = embeddings.cpu().numpy()
                    all_embeddings.append(embeddings)
                
                pbar.update(len(batch))
            pbar.close()
            return np.vstack(all_embeddings)
        else:
            # Standard SentenceTransformer encode
            # BGE models often use prompts/instructions too
            # We'll use a simple approach for the POC
            return self.model.encode(
                texts, 
                batch_size=batch_size, 
                show_progress_bar=True,
                convert_to_numpy=True
            )

class EnhancedQueryInteraction(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Project Query into Graph Space before Attention
        self.query_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=4,
            batch_first=True
        )
        self.relevance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) 
        )

    def forward(self, entity_reps, query_emb):
        """
        entity_reps: [Num_Entities, Hidden_Dim] (Graph Nodes)
        query_emb:   [Batch, Hidden_Dim] or [Hidden_Dim]
        """
        # 1. Align/Project Query
        q_aligned = self.query_projector(query_emb) # Output: [Batch, Dim] or [Dim]
        
        # 2. Shape Handling for Attention [Batch, Seq_Len, Dim]
        
        # Handle Q (Entities) -> Treat as Sequence of length Num_Entities, Batch size 1
        # Input: [Num_Entities, Dim] -> Output: [1, Num_Entities, Dim]
        if entity_reps.dim() == 2:
            q = entity_reps.unsqueeze(0) 
        else:
            q = entity_reps # Already batched?

        # Handle KV (Query) -> Treat as Sequence of length 1
        # We need shape: [1, 1, Dim]
        
        if q_aligned.dim() == 1: 
            # If input was [Dim] -> [1, 1, Dim]
            kv = q_aligned.unsqueeze(0).unsqueeze(1)
        elif q_aligned.dim() == 2:
            # If input was [Batch/1, Dim] -> [Batch/1, 1, Dim]
            kv = q_aligned.unsqueeze(1)
        else:
            # Already 3D?
            kv = q_aligned

        # 3. Cross Attention
        # Q = Graph Nodes, K/V = Query
        attn_output, _ = self.cross_attention(q, kv, kv)
        
        # Remove batch dimension to match original entity_reps shape
        # [1, Num_Entities, Dim] -> [Num_Entities, Dim]
        h_ent_aware = attn_output.squeeze(0)
        
        # 4. Compute Relevance Scores
        relevance_logits = self.relevance_head(h_ent_aware).squeeze(-1)
        
        return torch.sigmoid(relevance_logits), h_ent_aware

# --- 2. AUDITABLE HYBRID GNN MODEL ---
class AuditableHybridGNN(PretrainableHeteroGNN):
    def __init__(self, metadata, hidden_dim, num_layers=2):
        # Initialize backbone
        super().__init__(metadata, hidden_dim, out_dim=hidden_dim, num_layers=num_layers)
        
        # Interaction Module
        self.query_interaction = EnhancedQueryInteraction(hidden_dim)
        self.passage_norm = nn.LayerNorm(hidden_dim)
        
        self.scoring_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def load_backbone(self, checkpoint_path, device='cpu', freeze=True):
        """
        Loads pre-trained backbone weights (HGT, SGFormer, Norm) and optionally freezes them.
        """
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Backbone checkpoint {checkpoint_path} not found.")
            return False
            
        print(f"Loading pretrained backbone from {checkpoint_path}...")
        pretrained_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_dict = self.state_dict()
        
        # Filter weights: match keys and shapes, exclude task-specific layers
        load_dict = {
            k: v for k, v in pretrained_dict.items() 
            if k in model_dict and v.size() == model_dict[k].size()
        }
        
        self.load_state_dict(load_dict, strict=False)
        print(f"Successfully loaded {len(load_dict)} backbone layers.")
        
        if freeze:
            # Freeze everything except the task-specific heads
            task_heads = ['relevance_mlp', 'scoring_head', 'passage_norm']
            frozen_count = 0
            for name, param in self.named_parameters():
                if not any(head in name for head in task_heads):
                    param.requires_grad = False
                    frozen_count += 1
            print(f"Frozen {frozen_count} backbone parameters. Task heads {task_heads} remain trainable.")
        return True

    def forward(self, x_dict, edge_index_dict, query_emb, return_logits=False):
        # Save Raw BGE Embeddings
        raw_passages = x_dict['passage']

        # [NEW] Feature Dropout (Training Only)
        # 50% chance to zero-out the Raw BGE signal during training.
        # This forces the model to use the Graph signal to solve the ranking.
        if self.training:
            mask = torch.rand_like(raw_passages) > 0.5
            raw_passages_aug = raw_passages * mask
        else:
            raw_passages_aug = raw_passages

        # ... (Rest of Backbone: HGT, Global Reasoner, Interaction) ...
        # (Keep your existing code for steps 1-4)
        
        # 1. Backbone: HGT
        h_dict = x_dict
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {k: F.gelu(v) for k, v in h_dict.items()}
        
        # 2. Backbone: Global Reasoner
        h_local_ent = h_dict['entity']
        batch_ent = h_local_ent.new_zeros(h_local_ent.size(0), dtype=torch.long)
        h_ent_global = self.entity_global_reasoner.trans_conv(h_local_ent, batch_ent)
        h_dict['entity'] = self.entity_norm((1 - self.alpha) * h_local_ent + self.alpha * h_ent_global)

        # 3. Interaction
        relevance, h_ent_aware = self.query_interaction(h_dict['entity'], query_emb)
        
        # 4. Message Passing
        e2s_idx, sent_idx = edge_index_dict[('entity', 'in', 'sentence')]
        weighted_ent = h_ent_aware[e2s_idx] * relevance[e2s_idx].unsqueeze(-1)
        sent_context = scatter(weighted_ent, sent_idx, dim=0, dim_size=h_dict['sentence'].size(0), reduce='sum')
        h_dict['sentence'] = F.gelu(h_dict['sentence'] + sent_context)
        
        s2p_idx, psg_idx = edge_index_dict[('sentence', 'in', 'passage')]
        psg_context = scatter(h_dict['sentence'][s2p_idx], psg_idx, dim=0, dim_size=h_dict['passage'].size(0), reduce='sum')
        h_dict['passage'] = self.passage_norm(h_dict['passage'] + psg_context)

        # 5. SCORING (GNN ONLY)
        graph_passages = h_dict['passage']
        
        # We concatenate Query + Graph_Passage (NO RAW PASSAGES)
        # Input dim for scoring_head must be changed back to: hidden_dim * 2
        if query_emb.dim() == 1:
            q_scoring = query_emb.unsqueeze(0).expand(graph_passages.size(0), -1)
        else:
            q_scoring = query_emb.expand(graph_passages.size(0), -1)
            
        # [CHANGE] Removed raw_passages from cat()
        combined_features = torch.cat([graph_passages, q_scoring], dim=-1)
        
        gnn_logits = self.scoring_head(combined_features).squeeze()

        if return_logits:
            return gnn_logits
        return torch.sigmoid(gnn_logits)

    def get_subgraph_guidance(self, x_dict, edge_index_dict, query_emb):
        """
        PRE-TRAINED STRUCTURAL GUIDANCE (No fine-tuning used).
        Computes HGT-style attention weights to guide the random walk.
        """
        self.eval()
        with torch.no_grad():
            if query_emb.dim() == 1:
                query_emb = query_emb.unsqueeze(0)
            
            # 1. Semantic Relevance (Restart Vector)
            # Find entities semantically close to the query
            q_norm = F.normalize(query_emb, p=2, dim=-1)
            e_norm = F.normalize(x_dict['entity'], p=2, dim=-1)
            relevance = torch.mm(e_norm, q_norm.t()).squeeze()
            relevance = F.relu(relevance)
            relevance = relevance / (relevance.sum() + 1e-9)

            # 2. Structural Transition Weights (HGT Attention Guidance)
            # Instead of uniform 1.0, we weight edges by how much the 
            # pre-trained GNN "cares" about them given the query.
            # We use the alignment between Query and Edge attributes (Relation embeddings).
            
            edge_type = ('entity', 're', 'entity')
            edge_weights = None
            if edge_type in edge_index_dict:
                edge_index = edge_index_dict[edge_type]
                # edge_attr in your builder stores relation embeddings
                edge_attr = getattr(x_dict, 'edge_attr_dict', {}).get(edge_type, None)
                if edge_attr is None:
                    # Fallback to looking in the edge_index_dict storage if it's there
                    # In HeteroData, it's often in global_graph[edge_type].edge_attr
                    pass 

            return relevance, None

# --- 3. KG BUILDER ---
def compute_hash(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()

class POCGraphBuilder:
    def __init__(self, embedding_model: NVEmbedV2EmbeddingModel, azure_config: dict, cache_path: str = "ie_cache.json"):
        self.embed_model = embedding_model
        self.ie_tool = SimpleAzureOpenIE(
            api_key=azure_config['api_key'],
            endpoint=azure_config['endpoint'],
            api_version=azure_config['api_version'],
            deployment_name=azure_config['deployment_name']
        )
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.entity_to_idx = {} 
        self.unique_entities = []
        
    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f, indent=2)

    # def extract_info(self, text: str):
    #     text_hash = md5(text.encode()).hexdigest()
    #     if text_hash in self.cache:
    #         return tuple(self.cache[text_hash])
        
    #     res = self.ie_tool.extract_info(text)
    #     self.cache[text_hash] = list(res)
    #     self._save_cache()
    #     return res
    def extract_info(self, text: str, auto_save: bool = False):
        """
        Extract info with caching. 
        auto_save is False by default to prevent slow-downs during large batch runs.
        """
        text_hash = md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            # The cache might store as a list, return as tuple to match extract_info return type
            res = self.cache[text_hash]
            return tuple(res) if isinstance(res, list) else res
        
        res = self.ie_tool.extract_info(text)
        self.cache[text_hash] = list(res)
        
        if auto_save:
            self._save_cache()
        return res
    def extract_query_anchors(self, query: str) -> List[str]:
        """Extract entities from the query to serve as anchor nodes."""
        # Re-use the Azure IE tool for entity extraction from the query
        _, entities, _, _ = self.ie_tool.extract_info(query)
        return entities

    def build_subgraph_from_anchors(self, global_graph: HeteroData, anchor_entities: List[str], query_emb: torch.Tensor = None, k_semantic: int = 20) -> HeteroData:
        """
        Hybrid Subgraph Extraction (O(N) for semantic search, O(local) for traversal):
        1. Anchors: Physical (IE) + Semantic (Embedding Similarity)
        2. Discovery: Anchors -> (Sentences + Passages)
        3. Expansion: Contextual "Pull-Back" for Global Attention
        """
        # --- 1. COLLECT HYBRID ANCHORS ---
        # Create a normalized lookup for the global entities
        norm_entity_map = {ent.lower().strip(): idx for ent, idx in self.entity_to_idx.items()}
        
        anchor_indices = []
        for ent in anchor_entities:
            norm_ent = ent.lower().strip()
            if norm_ent in norm_entity_map:
                anchor_indices.append(norm_entity_map[norm_ent])
        print(f"Anchor indices: {anchor_indices}")
        physical_subset = torch.tensor(anchor_indices, device=global_graph['entity'].x.device) if anchor_indices else torch.tensor([], dtype=torch.long)

        # B. Semantic Anchors (Top-K semantic matches in Global Graph)
        # This is the O(N) step that ensures Global Attention has "seeds" outside the physical path
        semantic_subset = torch.tensor([], dtype=torch.long)
        if query_emb is not None:
            # Move query to same device as graph for fast O(N) matmul
            q_norm = F.normalize(query_emb.view(1, -1), p=2, dim=-1)
            e_norm = F.normalize(global_graph['entity'].x, p=2, dim=-1)
            sims = torch.matmul(e_norm, q_norm.t()).squeeze()
            semantic_subset = torch.topk(sims, k=min(k_semantic, sims.size(0))).indices

        # Combine Anchors
        anchor_subset = torch.unique(torch.cat([physical_subset, semantic_subset]))
        if anchor_subset.numel() == 0:
            return None

        # --- 2. DIRECT RELATION DISCOVERY (Entity -> Entity) ---
        # This ensures entities connected via 're' are included even if they aren't in the same sentence
        e2e_idx = global_graph['entity', 're', 'entity'].edge_index
        # Find neighbors of anchors
        mask_re = torch.isin(e2e_idx[0], anchor_subset)
        neighbors_re = e2e_idx[1][mask_re]
        
        # --- 3. DUAL-PATH DISCOVERY (E -> S, E -> P) ---
        e2s_idx = global_graph['entity', 'in', 'sentence'].edge_index
        sent_mask = torch.isin(e2s_idx[0], anchor_subset)
        sentence_subset = torch.unique(e2s_idx[1][sent_mask])
        
        e2p_idx = global_graph['entity', 'in', 'passage'].edge_index
        psg_mask_direct = torch.isin(e2p_idx[0], anchor_subset)
        passage_subset_direct = e2p_idx[1][psg_mask_direct]
        
        s2p_idx = global_graph['sentence', 'in', 'passage'].edge_index
        psg_mask_from_sent = torch.isin(s2p_idx[0], sentence_subset)
        passage_subset = torch.unique(torch.cat([passage_subset_direct, s2p_idx[1][psg_mask_from_sent]]))
        
        # --- 4. CONTEXTUAL EXPANSION (The "Pull-Back") ---
        s2e_idx = global_graph['sentence', 'hv', 'entity'].edge_index
        ent_from_sent = s2e_idx[1][torch.isin(s2e_idx[0], sentence_subset)]
        
        p2e_idx = global_graph['passage', 'hv', 'entity'].edge_index
        ent_from_psg = p2e_idx[1][torch.isin(p2e_idx[0], passage_subset)]
        
        # Combine everything into the final entity set
        final_entity_subset = torch.unique(torch.cat([
            anchor_subset, 
            neighbors_re,    # Added direct relations
            ent_from_sent, 
            ent_from_psg
        ]))
        
        # 5. Final Subgraph Conversion
        node_subset_dict = {
            'entity': final_entity_subset,
            'sentence': sentence_subset,
            'passage': passage_subset
        }
        subgraph = global_graph.subgraph(node_subset_dict)
        
        # Map back to global IDs
        for node_type, subset in node_subset_dict.items():
            subgraph[node_type].n_id = subset
            
        return subgraph

    def build_attention_guided_subgraph(self, global_graph: HeteroData, anchor_entities: List[str], query_emb: torch.Tensor, k_top: int = 300, steps: int = 3):
        """
        Variant: Random Walk guided by Hybrid Attention (Local edges + Global semantic bias)
        """
        num_entities = global_graph['entity'].num_nodes
        device = global_graph['entity'].x.device
        
        # --- 1. GLOBAL ATTENTION MAP (Guidance Field) ---
        q_norm = F.normalize(query_emb.view(1, -1), p=2, dim=-1)
        e_norm = F.normalize(global_graph['entity'].x, p=2, dim=-1)
        global_relevance = torch.matmul(e_norm, q_norm.t()).squeeze()
        global_relevance = torch.clamp(global_relevance, min=0) 
        
        # --- 2. INITIALIZE LOCAL SEEDS ---
        relevance = torch.zeros(num_entities, device=device)
        anchors = [self.entity_to_idx[e] for e in anchor_entities if e in self.entity_to_idx]
        if not anchors:
            return None
        relevance[anchors] = 1.0 
        
        # --- 3. ATTENTION-WEIGHTED PROPAGATION ---
        e2e_idx = global_graph['entity', 're', 'entity'].edge_index
        e2s_idx = global_graph['entity', 'in', 'sentence'].edge_index
        s2e_idx = global_graph['sentence', 'hv', 'entity'].edge_index

        for _ in range(steps):
            # A. Flow through E-E relations with Global Bias
            msg_e2e = relevance[e2e_idx[0]] * global_relevance[e2e_idx[1]]
            new_rel_e2e = scatter(msg_e2e, e2e_idx[1], dim=0, dim_size=num_entities, reduce='sum')
            
            # B. Flow through Sentences (E -> S -> E)
            sent_relevance = scatter(relevance[e2s_idx[0]], e2s_idx[1], dim=0, reduce='sum')
            msg_s2e = sent_relevance[s2e_idx[0]] * global_relevance[s2e_idx[1]]
            new_rel_s2e = scatter(msg_s2e, s2e_idx[1], dim=0, dim_size=num_entities, reduce='sum')
            
            relevance = relevance + new_rel_e2e + new_rel_s2e
            relevance = F.normalize(relevance, p=1, dim=0)

        # --- 4. SELECTION ---
        # Combine Local walk and Global direct similarity
        final_scores = relevance + (0.1 * global_relevance) 
        top_entity_indices = torch.topk(final_scores, k=min(k_top, num_entities)).indices

        # --- 5. DISCOVER CONTEXTUAL NODES (S and P) ---
        # Find sentences where these top entities appear
        sent_mask = torch.isin(e2s_idx[0], top_entity_indices)
        sentence_subset = torch.unique(e2s_idx[1][sent_mask])
        
        # Find passages (Directly from entities or via sentences)
        e2p_idx = global_graph['entity', 'in', 'passage'].edge_index
        psg_mask_direct = torch.isin(e2p_idx[0], top_entity_indices)
        passage_subset_direct = e2p_idx[1][psg_mask_direct]
        
        s2p_idx = global_graph['sentence', 'in', 'passage'].edge_index
        psg_mask_from_sent = torch.isin(s2p_idx[0], sentence_subset)
        passage_subset = torch.unique(torch.cat([passage_subset_direct, s2p_idx[1][psg_mask_from_sent]]))
        
        # --- 6. FINAL PULL-BACK (Contextual Completion) ---
        # Ensure we have all entities found in the retrieved context
        ent_from_sent = s2e_idx[1][torch.isin(s2e_idx[0], sentence_subset)]
        p2e_idx = global_graph['passage', 'hv', 'entity'].edge_index
        ent_from_psg = p2e_idx[1][torch.isin(p2e_idx[0], passage_subset)]
        
        final_entity_subset = torch.unique(torch.cat([top_entity_indices, ent_from_sent, ent_from_psg]))
        
        # --- 7. SUBGRAPH CONVERSION ---
        node_subset_dict = {
            'entity': final_entity_subset,
            'sentence': sentence_subset,
            'passage': passage_subset
        }
        subgraph = global_graph.subgraph(node_subset_dict)
        
        # Attach global IDs (n_id) for traceability in EvalGNN.py
        for node_type, subset in node_subset_dict.items():
            subgraph[node_type].n_id = subset
            
        # Optional: Print entities for debugging
        entity_texts = [self.unique_entities[i] for i in final_entity_subset.tolist()]
        print(f"\n--- Attention Subgraph Entities ({len(entity_texts)}) ---")
            
        return subgraph
    # Extract Subgraph based on these high-attention nodes
    # (Same as your previous subgraph selection logic)
    # ... build final subgraph using these subsets ...   
    # def extract_info_old(self, text: str):
    #     # SIMPLIFIED: Using regex for POC. In production, use the LLM-based OpenIE.
    #     # This mocks extracting entities (capitalized words) and simple sentences.
    #     sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 10]
    #     entities = list(set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text)))
        
    #     # Simplified triples: (Entity, "related_to", Sentence_Snippet)
    #     triples = []
    #     for ent in entities:
    #         for i, sent in enumerate(sentences):
    #             if ent in sent:
    #                 triples.append([ent, "appears_in", f"sentence_{i}"])
    #     return sentences, entities, triples
    
    #def extract_info(self, text: str, chunk_id: str = "poc_chunk"):
    #    # This now calls our simplified Azure-based method
    #   return self.ie_tool.extract_info(text)
        
    def build_graph(self, docs: List[str]):
        data = HeteroData()
        all_sentences_info = [] # Stores global sentence text and doc mapping
        all_doc_triples = []
        
        # 1. IDENTIFY MISSING CHUNKS (Optimized for unique hashes)
        doc_hashes = [md5(d.encode()).hexdigest() for d in docs]
        
        unique_missing = {}
        for idx, h in enumerate(doc_hashes):
            if h not in self.cache and h not in unique_missing:
                unique_missing[h] = idx
        
        #unique_missing={} # debug setting to empty array for now.
        # 2. BATCH PROCESS MISSING CHUNKS
        if unique_missing:
            print(f"Processing {len(unique_missing)} unique new chunks through OpenIE...")
            for h, idx in tqdm(unique_missing.items(), desc="Extracting Knowledge"):
                res = self.ie_tool.extract_info(docs[idx])
                
                # CHECK: Only cache if we actually got entities or triples back
                if res[1] or res[2]: 
                    self.cache[h] = {
                        "passage": docs[idx],
                        "sentences": res[0],
                        "entities": res[1],
                        "triples": res[2],
                        "sentence_entities": res[3]
                    }
                else:
                    print(f"Skipping cache for hash {h} due to empty extraction (will retry next run).")
                
                if len(self.cache) % 100 == 0:
                    self._save_cache()

            self._save_cache()

        print("Extracting OpenIE info...",len(docs))
        # 3. CONSTRUCT LOCAL GRAPH DATA FROM CACHE
        for i, doc_hash in enumerate(doc_hashes):
            if doc_hash not in self.cache:
                continue
            cached_res = self.cache[doc_hash]
            
            sents = cached_res["sentences"]
            triples = cached_res["triples"]
            sent_ents = cached_res["sentence_entities"]
            
            for s_idx, s_text in enumerate(sents):
                all_sentences_info.append({
                    'text': s_text,
                    'doc_idx': i,
                    'entities': sent_ents[s_idx]
                })
            all_doc_triples.extend(triples)
            
        print("all triples: ", len(all_doc_triples))
        #print(all_doc_triples)
        all_doc_triples = [t for t in all_doc_triples if isinstance(t, list) and len(t) == 3]
        print("after triples: ", len(all_doc_triples))
        # 1. Resolve Unique Entities
        all_entities_set = set()
        for t in all_doc_triples:
            # Ensure sub and obj are strings (LLMs sometimes return lists or None)
            sub = t[0] if isinstance(t[0], str) else str(t[0])
            obj = t[2] if isinstance(t[2], str) else str(t[2])
            all_entities_set.add(sub)
            all_entities_set.add(obj)
            
        for s_info in all_sentences_info:
            for ent in s_info['entities']:
                if isinstance(ent, str):
                    all_entities_set.add(ent)
                else:
                    all_entities_set.add(str(ent))
            
        unique_entities = sorted(list(all_entities_set))
        self.unique_entities = unique_entities
        self.entity_to_idx = {ent: i for i, ent in enumerate(unique_entities)}
        ent_to_idx = self.entity_to_idx
        print("unique_entities: ", len(unique_entities))

        # 2. Generate Node Embeddings
        print("Generating Node Embeddings...")
        data['passage'].x = torch.from_numpy(self.embed_model.batch_encode(docs, instruction="passage"))
        data['sentence'].x = torch.from_numpy(self.embed_model.batch_encode([s['text'] for s in all_sentences_info], instruction="sentence"))
        data['entity'].x = torch.from_numpy(self.embed_model.batch_encode(unique_entities, instruction="entity"))
        
        # 3. Structural Relation Embeddings (for edge_attr)
        structural_rels = ["in", "hv", "seq"]
        struct_rel_vecs = self.embed_model.batch_encode(structural_rels, instruction="relation")
        structural_rel_emb = {rel: torch.from_numpy(vec) for rel, vec in zip(structural_rels, struct_rel_vecs)}

        # 4. Build Edges
        print("Building Edges...")
        
        # A. Entity -> Entity (from Triples + Inverses)
        e2e_edges, e2e_relation_texts = [], []
        for t in all_doc_triples:
            # Re-sanitize to match keys in ent_to_idx
            sub = t[0] if isinstance(t[0], str) else str(t[0])
            rel = t[1] if isinstance(t[1], str) else str(t[1])
            obj = t[2] if isinstance(t[2], str) else str(t[2])
            
            s_idx, o_idx = ent_to_idx[sub], ent_to_idx[obj]
            # Forward
            e2e_edges.append([s_idx, o_idx])
            e2e_relation_texts.append(rel)
            # Inverse (Reasoning Path)
            e2e_edges.append([o_idx, s_idx])
            e2e_relation_texts.append(f"inverse {rel}")
            
        if e2e_edges:
            data['entity', 're', 'entity'].edge_index = torch.tensor(e2e_edges).t().contiguous()
            data['entity', 're', 'entity'].edge_attr = torch.from_numpy(
                self.embed_model.batch_encode(e2e_relation_texts, instruction="relation")
            )

        # B. Sentence <-> Passage (Hierarchical)
        s2p_edges = [[i, s['doc_idx']] for i, s in enumerate(all_sentences_info)]
        data['sentence', 'in', 'passage'].edge_index = torch.tensor(s2p_edges).t().contiguous()
        data['passage', 'hv', 'sentence'].edge_index = data['sentence', 'in', 'passage'].edge_index.flip(0)
        
        data['sentence', 'in', 'passage'].edge_attr = structural_rel_emb["in"].repeat(len(s2p_edges), 1)
        data['passage', 'hv', 'sentence'].edge_attr = structural_rel_emb["hv"].repeat(len(s2p_edges), 1)

        # C. Entity <-> Sentence
        e2s_edges = []
        for s_idx, s_info in enumerate(all_sentences_info):
            for ent in s_info['entities']:
                # Re-sanitize to match keys in ent_to_idx
                ent_str = ent if isinstance(ent, str) else str(ent)
                if ent_str in ent_to_idx:
                    e2s_edges.append([ent_to_idx[ent_str], s_idx])
        
        if e2s_edges:
            data['entity', 'in', 'sentence'].edge_index = torch.tensor(e2s_edges).t().contiguous()
            data['sentence', 'hv', 'entity'].edge_index = data['entity', 'in', 'sentence'].edge_index.flip(0)
            data['entity', 'in', 'sentence'].edge_attr = structural_rel_emb["in"].repeat(len(e2s_edges), 1)
            data['sentence', 'hv', 'entity'].edge_attr = structural_rel_emb["hv"].repeat(len(e2s_edges), 1)

        # D. Entity <-> Passage
        e2p_edges = []
        doc_entities = [set() for _ in range(len(docs))]
        for s_info in all_sentences_info:
            doc_entities[s_info['doc_idx']].update(s_info['entities'])
        
        for d_idx, ents in enumerate(doc_entities):
            for ent in ents:
                e2p_edges.append([ent_to_idx[ent], d_idx])
                
        if e2p_edges:
            data['entity', 'in', 'passage'].edge_index = torch.tensor(e2p_edges).t().contiguous()
            data['passage', 'hv', 'entity'].edge_index = data['entity', 'in', 'passage'].edge_index.flip(0)
            data['entity', 'in', 'passage'].edge_attr = structural_rel_emb["in"].repeat(len(e2p_edges), 1)
            data['passage', 'hv', 'entity'].edge_attr = structural_rel_emb["hv"].repeat(len(e2p_edges), 1)

        # E. Sentence -> Sentence (Sequential Flow ONLY)
        s2s_edges = []
        for d_idx in range(len(docs)):
            # Get sentences for this doc in ORDER
            doc_sents = [i for i, s in enumerate(all_sentences_info) if s['doc_idx'] == d_idx]
            
            # Connect s[i] <-> s[i+1] (Chain, not Clique)
            for k in range(len(doc_sents) - 1):
                u, v = doc_sents[k], doc_sents[k+1]
                
                # Forward (Next)
                s2s_edges.append([u, v])
                # Backward (Prev) - Optional, but good for GNN flow
                s2s_edges.append([v, u])
        
        if s2s_edges:
            data['sentence', 're', 'sentence'].edge_index = torch.tensor(s2s_edges).t().contiguous()
            data['sentence', 're', 'sentence'].edge_attr = structural_rel_emb["seq"].repeat(len(s2s_edges), 1)

        print(f"Nodes: {len(docs)} Passages, {len(all_sentences_info)} Sentences, {len(unique_entities)} Entities")
        return data, unique_entities

# --- 4. DATA LOADING ---
def load_musique_samples(path: str, limit: int = 10):
    with open(path, "r") as f:
        data = json.load(f)
    return data[:limit]

def load_musique_corpus(path: str, limit: int = 100):
    with open(path, "r") as f:
        data = json.load(f)
    # Reformat to match the docs list format [title \n text]
    return [f"{doc['title']}\n{doc['text']}" for doc in data[:limit]]

# --- 4. DATA LOADING & EVALUATION ---
def load_musique_samples(path: str, limit: int = 10):
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Returning empty list.")
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data[:limit]

def evaluate_with_subgraphs(model, builder, samples, global_graph, embed_model, k_list=[1, 5]):
    model.eval()
    recalls = {k: [] for k in k_list}
    
    print(f"Evaluating with Subgraph Extraction on {len(samples)} samples...")
    with torch.no_grad():
        for sample in samples:
            question = sample['question']
            
            # Step 1: Identify Anchor Entities in the Query
            anchors = builder.extract_query_anchors(question)
            
            # Step 2: Extract Subgraph (Algorithm 5)
            subgraph = builder.build_subgraph_from_anchors(global_graph, anchors)
            
            if subgraph is None:
                for k in k_list: recalls[k].append(0.0)
                continue
                
            # Step 3: Get Query Embedding
            query_emb = torch.from_numpy(embed_model.batch_encode([question], instruction="query"))
            
            # Step 4: GNN Forward Pass on Subgraph
            scores = model(subgraph.x_dict, subgraph.edge_index_dict, query_emb)
            
            # Step 5: Map Subgraph Scores back to Global Passage Indices
            # (HeteroData.subgraph stores original indices in subgraph[node_type].tf_index or similar, 
            # but in recent PyG it is subgraph[node_type].n_id)
            psg_indices = subgraph['passage'].n_id
            
            # Find which of the supporting paragraphs are in the subgraph
            # This requires knowing the global indices of the supporting paragraphs.
            # In MuSiQue, we'd need to map supporting paragraph IDs to our global_graph indices.
            
            # (Note: This is a sketch. Actual mapping depends on how global_graph was built)
            pass 

    return recalls


