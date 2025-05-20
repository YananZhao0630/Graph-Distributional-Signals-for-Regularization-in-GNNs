import argparse
import time
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
    def __init__(self, in_size, hid_size, out_size, heads):
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
            h = layer(g, h)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h

def p_loss(g, x):
    #implementation of the p-loss defined in https://arxiv.org/pdf/2009.02027.pdf
    device = g.device
    num_nodes = g.number_of_nodes()
    degs = g.in_degrees().float()
    deg_mat = torch.diag(degs)  # get degree matrix to calculate laplacian
    adj = g.adj_external(scipy_fmt='csr').astype(float)
    #adj = adj.tocoo()
    #adj = torch.sparse.FloatTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
    #                               torch.FloatTensor(adj.data.astype(np.float64)))
    # adj = g.adjacency_matrix(scipy_fmt="csr")
    adj = torch.from_numpy(adj.toarray()).float().to(device)
    normalize_adj = torch.matmul(torch.inverse(deg_mat), adj)
    y = torch.matmul(normalize_adj, x.float())
    z = (torch.linalg.matrix_norm(y-x.float()))**2
    return z/num_nodes

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
        start_time = time.time()
        model.train()
        logits = model(g, features)
        eta = 0.0005
        loss1 = loss_fcn(logits[train_mask], labels[train_mask])
        loss2 = p_loss(g, logits)
        loss = loss1+ eta*loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end_time = time.time()

        # Calculate and print the total execution time
        total_time = end_time - start_time
        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time {:.4f}  ".format(
                epoch, loss.item(), acc, total_time
            )
        )


if __name__ == "__main__":
    # start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed').",
    )
    args = parser.parse_args()
    # args.dataset = "pubmed"
    # print(f"Training with DGL built-in GATConv module.")

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
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    # create GAT model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GAT(in_size, 8, out_size, heads=[8, 1]).to(device)

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
    print("Test accuracy mean {:.4f}".format(acc_mean))
    print("Test accuracy variation {:.4f}".format(acc_var))


    # print(f"Total running time: {total_time:.4f} seconds")