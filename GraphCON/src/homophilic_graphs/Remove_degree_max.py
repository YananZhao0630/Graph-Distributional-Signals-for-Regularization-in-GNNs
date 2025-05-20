import argparse
import os
import numpy as np
import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.datasets import Planetoid, WebKB
import torch.nn.functional as F
from GNN import GNN
from GNN_early import GNNEarly
import time
from data import get_dataset, set_train_val_test_split
from ogb.nodeproppred import Evaluator
from good_params_graphCON import good_params_dict
from torch_geometric.utils import degree
import wandb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed', 'wisconsin', 'texas', 'cornell','chameleon').")

    args = parser.parse_args()

    if args.dataset == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0]
    elif args.dataset == 'texas':
        dataset = WebKB(root='/tmp/Texas', name='Texas')
        data = dataset[0]

    # Adjacency matrix
    adj_matrix = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)), (data.num_nodes, data.num_nodes)).to_dense()
    removed_node_count = 0

    # Loop to iteratively remove the node with the maximum degree
    while adj_matrix.sum() > 0:  # Continue until the adjacency matrix becomes all zeros
        # Calculate the degree of each node
        degrees = adj_matrix.sum(dim=1)

        # Find the node with the maximum degree
        max_degree_node = torch.argmax(degrees)

        # Remove the node with the maximum degree (set its row and column to zero)
        adj_matrix[max_degree_node, :] = 0
        adj_matrix[:, max_degree_node] = 0

        # Increment the removed node count
        removed_node_count += 1

        # Optional: Print the remaining number of edges
        print(f'Removed node {max_degree_node.item()}, remaining edges: {adj_matrix.sum().item()}')

    # Print the total number of removed nodes
    print(f"All nodes removed, adjacency matrix is zero. Total removed nodes: {removed_node_count}")