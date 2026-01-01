import torch
from torch_geometric.loader import HGTLoader
from torch_geometric.data import HeteroData

def test_loader():
    path = '/home/ubuntu/gnn-hgt/GNN-HGT-KG-AUDIT/kg_storage/global_graph_bge_small.pt'
    data = torch.load(path, map_location='cpu', weights_only=False)
    
    batch_size = 8
    loader = HGTLoader(
        data,
        num_samples=[100, 50],
        batch_size=batch_size,
        input_nodes='entity',
    )
    
    batch = next(iter(loader))
    print(f"Batch: {batch}")
    print(f"Entity nodes in batch: {batch['entity'].x.shape[0]}")
    # Check if seed nodes are at the beginning
    # In NodeLoader, the first batch_size nodes of the input node type are the seed nodes.
    print(f"Seed nodes should be the first {batch_size} entities.")
    
    # Check edge types sampled
    print(f"Edge types in batch: {batch.edge_types}")
    for etype in batch.edge_types:
        print(f"  {etype}: {batch[etype].edge_index.shape[1]} edges")

if __name__ == "__main__":
    test_loader()

