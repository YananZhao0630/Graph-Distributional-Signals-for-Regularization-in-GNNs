import argparse

import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
import scipy
from scipy import sparse
from scipy.io import savemat

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, number_layers, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(
            dglnn.GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        for _ in range(number_layers-2):
            self.gat_layers.append(
                dglnn.GATConv(
                    hid_size * heads[0],
                    hid_size,
                    heads[1],
                    feat_drop=0.6,
                    attn_drop=0.6,
                    activation=None,
                )
            )
        self.gat_layers.append(
            dglnn.GATConv(
                hid_size * heads[0],
                out_size,
                heads[1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            )
        )
    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            if i > 0:
                h_old = h
            h = layer(g, h)
            if i == 127:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                if i > 0:
                    h = h.flatten(1)+ h_old
                else:
                    h = h.flatten(1)
        return h

def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

    # training loop
    for epoch in range(200):
        model.train()
        logits = model(g, features)
        loss1 = loss_fcn(logits[train_mask], labels[train_mask])
        loss = loss1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}".format(
                epoch, loss.item(), acc
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed').",
    )
    args = parser.parse_args()
    args.dataset = "cora"
    print("Training with DGL built-in GATConv module for dataset {}".format("pubmed"))

    # load and preprocess dataset
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    g = data[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    # create GAT model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GAT(in_size, 8, out_size, number_layers=128, heads=[8, 8]).to(device)

    # model training
    acc_list = []
    for Ex_time in range(10):
        print("Training...")
        train(g, features, labels, masks, model)

        # test the model
        print("Testing...")
        acc = evaluate(g, features, labels, masks[2], model)
        print("Test accuracy {:.4f}".format(acc))
        acc_list.append(acc)

    print(acc_list)
    acc_mean = np.mean(acc_list)
    acc_var = np.std(acc_list)
    print("Test accuracy mean {:.4f}".format(acc_mean*100))
    print("Test accuracy variation {:.4f}".format(acc_var*100))
