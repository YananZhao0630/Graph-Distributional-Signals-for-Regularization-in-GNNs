import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, AmazonCoBuyPhotoDataset, CoauthorCSDataset
from dgl import AddSelfLoop
import scipy
from scipy import sparse
from scipy.io import savemat
import argparse
import numpy as np
import scipy.sparse as sp
import time

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)
# g: representing the input graph; features representing the input node features
    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item()*1.0 / len(labels)

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

def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0] # Extacts the train and validation masks from the 'masks' list
    val_mask = masks[1] # masks seperate the data into training and validation subsets
    loss_fcn = nn.CrossEntropyLoss() # defines the loss function as cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    # Initializes the Adam optimizer with the model parameters as the optimization variables.
    # training loop
    for epoch in range(80):
        start_time = time.time()
        model.train()  # set the model in the training mode
        logits = model(g, features)  # computes the logits (raw model outputs)
        alpha = 0.001
        belta = 0.001
        m = 0.1
        eta = 0.01
        loss1 = loss_fcn(logits[train_mask], labels[train_mask])
        loss2 = LEReg_loss(g,logits, alpha, belta, m)
        loss = loss1 + eta*loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end_time = time.time()
        # Calculate and print the total execution time
        total_time = end_time - start_time
        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | time {:.4f}".format(
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
    # args.dataset = 'pubmed'
    # print(f"Training with DGL built-in GraphConv module.")

    # load and precess dataset
    transform = (
        AddSelfLoop()
    ) # by default, it will first remove self-loops to prevent duplication
    if args.dataset == "cora":
         data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
         data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
         data = PubmedGraphDataset(transform=transform)
    else:
         raise ValueError("Unknown dataset: {}".format(args.dataset))

    g = data[0]
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
    # create GCN model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GCN(in_size, 16, out_size).to(device)

    # model training

    acc_list = []
    for Ex_time in range(10):
        print("Training...")
        train(g, features, labels, masks, model)

        # model testing
        print("Testing....")
        acc = evaluate(g, features, labels, masks[2], model)
        acc_list.append(acc)
        print("Test accuracy {:.4f}".format(acc))

    print(acc_list)
    acc_mean = np.mean(acc_list)
    acc_var = np.std(acc_list)
    print("Test accuracy mean {:.4f}".format(acc_mean))
    print("Test accuracy variation {:.4f}".format(acc_var))


    # End the timer

    # print(f"Total running time: {total_time:.4f} seconds")


