import argparse
import os
import numpy as np
import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import CoauthorCSDataset, AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset, CoauthorPhysicsDataset
import torch.nn.functional as F
from GNN import GNN
from GNN_early import GNNEarly
import time
from data import get_dataset, set_train_val_test_split
from ogb.nodeproppred import Evaluator
from good_params_graphCON import good_params_dict
from dgl import AddSelfLoop
from scipy import sparse
import wandb

