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

# --- 2. AUDITABLE HYBRID GNN MODEL ---
class AuditableHybridGNN(nn.Module):
    def __init__(self, metadata, hidden_dim):
        super().__init__()
        self.local_hgt = HGTConv(hidden_dim, hidden_dim, metadata, heads=4)
        #self.entity_global_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.entity_global_reasoner = SGFormer(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            trans_num_layers=1,    # Number of global attention layers
            trans_num_heads=4,     # Number of attention heads
            gnn_num_layers=0,      # We use HGT for local, so disable SGFormer's internal GNN
            graph_weight=0.0       # Only use the global transformer output
        )
        self.entity_norm = nn.LayerNorm(hidden_dim)
        self.passage_norm = nn.LayerNorm(hidden_dim)
        self.alpha = 0.1
        self.scoring_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_dict, edge_index_dict, query_emb):
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
            
        # 1. Local Structural Reasoning (HGT)
        h_dict = self.local_hgt(x_dict, edge_index_dict)
        
        # 2. Global Entity Reasoning (Self-Attention among entities)
        h_local_ent = h_dict['entity']
        # Create the missing batch vector for SGFormer (all nodes in one graph instance)
        batch_ent = h_local_ent.new_zeros(h_local_ent.size(0), dtype=torch.long)
        
        # Call .trans_conv directly to get RAW features (no log_softmax)
        # This mirrors the logic in the pre-training script for high-quality features
        h_ent_global = self.entity_global_reasoner.trans_conv(h_local_ent, batch_ent)
        
        # Residual connection and normalization
        h_dict['entity'] = self.entity_norm((1 - self.alpha) * h_local_ent + self.alpha * h_ent_global)
        
        # 3. Query-Guided Broadcast (Update Passages based on Query-Relevant Entities)
        e2p_index = edge_index_dict[('entity', 'in', 'passage')]
        ent_idx, psg_idx = e2p_index

        # Calculate relevance of each entity to the query
        q_expanded = query_emb.expand(h_dict['entity'].size(0), -1)
        relevance = torch.sum(h_dict['entity'] * q_expanded, dim=-1).sigmoid()

        # Weight entity features by relevance
        weighted_ent_features = h_dict['entity'][ent_idx] * relevance[ent_idx].unsqueeze(-1)

        # Aggregate weighted features into passages
        psg_context = scatter(src=weighted_ent_features, 
                            index=psg_idx, 
                            dim=0, 
                            dim_size=h_dict['passage'].size(0), 
                            reduce='sum')

        h_dict['passage'] = self.passage_norm(h_dict['passage'] + psg_context)

        # 4. Final Scoring: Passage Embedding + Query Embedding
        passages = h_dict['passage']
        q_scoring = query_emb.expand(passages.size(0), -1)
        
        scores = self.scoring_head(torch.cat([passages, q_scoring], dim=-1)).squeeze()
        return scores

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
        # The cache will now store full objects: { "hash": {"passage": "...", "triples": [...], "entities": [...] } }
        self.cache = self._load_cache()
        
    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def extract_info(self, text: str):
        text_hash = md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            return tuple(self.cache[text_hash])
        
        res = self.ie_tool.extract_info(text)
        self.cache[text_hash] = list(res)
        self._save_cache()
        return res
        
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
        
        unique_missing={} # debug setting to empty array for now.
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
        ent_to_idx = {ent: i for i, ent in enumerate(unique_entities)}
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

        # E. Sentence -> Sentence (Intra-passage flow)
        s2s_edges = []
        for d_idx in range(len(docs)):
            doc_sents = [i for i, s in enumerate(all_sentences_info) if s['doc_idx'] == d_idx]
            for i in doc_sents:
                for j in doc_sents:
                    if i != j: s2s_edges.append([i, j])
        
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

def evaluate_retrieval(model, builder, samples, embed_model, k_list=[1, 5]):
    model.eval()
    recalls = {k: [] for k in k_list}
    
    print(f"Evaluating on {len(samples)} samples...")
    with torch.no_grad():
        for sample in samples:
            question = sample['question']
            #doc_texts = [f"{p['title']}\n{p['paragraph_text']}" for p in sample['paragraphs']]
            doc_texts = [f"{p['paragraph_text']}" for p in sample['paragraphs']]

            target_indices = [idx for idx, p in enumerate(sample['paragraphs']) if p['is_supporting']]
            
            graph, _ = builder.build_graph(doc_texts)
            query_emb = torch.from_numpy(embed_model.batch_encode([question], instruction="query"))
            
            scores = model(graph.x_dict, graph.edge_index_dict, query_emb)
            # Handle single doc edge case
            if scores.dim() == 0: scores = scores.unsqueeze(0)
            
            # Sort indices by score descending
            top_indices = torch.argsort(scores, descending=True).tolist()
            
            for k in k_list:
                top_k = top_indices[:k]
                hit = any(idx in top_k for idx in target_indices)
                recalls[k].append(1.0 if hit else 0.0)
                
    results = {f"Recall@{k}": np.mean(v) for k, v in recalls.items()}
    return results
# --- 5. MAIN POC FLOW ---
if __name__ == "__main__":
    # 1. Configuration
    azure_config = {
        "api_key": "db8dac9cea3148d48c348ed46e9bfb2d",
        "endpoint": "https://bodeu-des-csv02.openai.azure.com/",
        "api_version": "2024-12-01-preview", 
        "deployment_name": "gpt-4o-mini" 
    }
    checkpoint_path = "auditable_gnn_poc.pth"

    # 2. Initialize Models
    embed_model = NVEmbedV2EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2") 
    builder = POCGraphBuilder(embed_model, azure_config, cache_path="musique_ie_cache.json")
    
    # Load Samples
    dataset_path = "dataset/musique_all.json"
    all_samples = load_musique_samples(dataset_path, limit=10)
    train_samples = all_samples[:5]
    eval_samples = all_samples[5:10]
    
    docs = load_musique_corpus("dataset/musique_corpus.json", limit=10)
    # 3. Model Setup
    graph, _ = builder.build_graph(docs)
    hidden_dim = graph['passage'].x.shape[1]
    model = AuditableHybridGNN(graph.metadata(), hidden_dim)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading existing model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # --- PHASE 1: ZERO-SHOT EVALUATION ---
    print("\n>>> Phase 1: Zero-Shot Evaluation")
    zs_results = evaluate_retrieval(model, builder, eval_samples, embed_model)
    print(f"Zero-Shot Results: {zs_results}")

    # --- PHASE 2: FINE-TUNING ---
    print(f"\n>>> Phase 2: Fine-Tuning on {len(train_samples)} samples...")
    for epoch in range(2):
        model.train()
        epoch_loss = 0
        for i, sample in enumerate(train_samples):
            question = sample['question']
            doc_texts = [f"{p['paragraph_text']}" for p in sample['paragraphs']]
            target_indices = [idx for idx, p in enumerate(sample['paragraphs']) if p['is_supporting']]
            if not target_indices: continue
            
            target_idx = torch.tensor([target_indices[0]]) # POC: take first supporting
            graph, _ = builder.build_graph(doc_texts)
            
            optimizer.zero_grad()
            query_emb = torch.from_numpy(embed_model.batch_encode([question], instruction="query"))
            scores = model(graph.x_dict, graph.edge_index_dict, query_emb)
            
            loss = criterion(scores.unsqueeze(0), target_idx)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch} | Avg Loss: {epoch_loss/len(train_samples):.4f}")
        # Save model after each epoch
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    # --- PHASE 3: POST-TRAIN EVALUATION ---
    print("\n>>> Phase 3: Post-Train Evaluation")
    final_results = evaluate_retrieval(model, builder, eval_samples, embed_model)
    print(f"Final Results: {final_results}")

    # 4. Optional: Build a larger KG from the Corpus
    # print("\nBuilding Global KG from Corpus (First 50 docs)...")
    # corpus_path = "dataset/musique_corpus.json"
    # corpus_docs = load_musique_corpus(corpus_path, limit=50)
    # global_graph, _ = builder.build_graph(corpus_docs)
    # print(f"Global KG Nodes: {global_graph['passage'].num_nodes} Passages")

