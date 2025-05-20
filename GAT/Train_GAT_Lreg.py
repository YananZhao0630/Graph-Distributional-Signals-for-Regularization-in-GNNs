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
import scipy.sparse as sp


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

def normalize_adj(A):
    # degree matrix
    # adj = A.toarray()  ## numpy array
    Dl = np.sum(A, 0)
    num_nodes = A.shape[0]
    Dn = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1/2)
    normal_A1 = np.dot(Dn, A)
    normal_A = np.dot(normal_A1,Dn)
    return normal_A

def LEReg_loss(g,x,m,a,b):
    P = F.softmax(x, dim=1)  # device: cuda 0; x the output of GNN
    adj = g.adj_external(scipy_fmt='csr').astype(float)
    adj = adj.toarray()
    norm_adj = normalize_adj(adj)
    norm_laplacian = sp.eye(adj.shape[0]) - norm_adj
    norm_laplacian = torch.from_numpy(norm_laplacian).float().to(device)  # norm_laplacian device: cuda 0

    # X'LX
    XT = torch.transpose(x, 0, 1)   # cuda 0
    XTLX = torch.mm(torch.mm(XT, norm_laplacian), x)  # cuda 0 ---> L_intra = tr(XTLX)

    PT = torch.transpose(P, 0, 1)  # cuda 0
    B = torch.mm(torch.mm(PT, torch.from_numpy(adj).float().to(device)), P) # cuda 0, tensor
    norm_B = normalize_adj(B.detach().cpu().numpy())
    L_B = sp.eye(norm_B.shape[0]) - norm_B  # csr matrix
    L_B = torch.from_numpy(L_B).float().to(device)  # 7x7
    # Y=P'X
    Y = torch.mm(PT, x)
    YT = torch.transpose(Y, 0, 1)
    # Y‘L_BY
    YTLBY = torch.mm(torch.mm(YT, L_B), Y)
    # print("second term", m-torch.trace(YTLBY))
    return a*torch.trace(XTLX)+b*max(0, m-torch.trace(YTLBY))

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
    for epoch in range(100):
        start_time = time.time()
        model.train()
        logits = model(g, features)
        alpha = 0.0005
        belta = 0.0005
        m = 0.1
        eta = 0.001
        loss1 = loss_fcn(logits[train_mask], labels[train_mask])
        loss2 = LEReg_loss(g, logits, alpha, belta, m)
        loss = loss1+ eta*loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end_time = time.time()
        # Calculate and print the total execution time
        total_time = end_time - start_time
        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | alpha={}, belta={}, m={}, eta={} | time {:.4f} ".format(
                epoch, loss.item(), acc, alpha, belta, m, eta, total_time
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
    # args.dataset = "pubmed"
    # print("Training with DGL built-in GATConv module for dataset {data_name}".format(data_name=args.dataset))

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
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
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