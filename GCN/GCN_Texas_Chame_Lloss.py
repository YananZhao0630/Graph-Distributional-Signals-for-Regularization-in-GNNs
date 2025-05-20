import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.data import TexasDataset, ChameleonDataset
from dgl import AddSelfLoop
import scipy
from scipy import sparse as sp
from scipy.io import savemat
import argparse
import numpy as np
from data_handling import get_data

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
        return correct.item()*1.0 / len(labels)

def train(g,features, labels, masks,model,eta, m ,a,b):
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    for epoch in range(500):
        model.train() # set the model in the training mode
        logits = model(g, features) # Computes the logits (raw model outputs)
        loss1 = loss_fn(logits[train_mask], labels[train_mask])
        loss = loss1 + eta*LEReg_loss(g, logits, m, a, b)
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
        default="chameleon",
        help="Dataset name ('texas', 'chameleon').",
    )
    args = parser.parse_args()
    args.dataset = 'chameleon'
    print(f"Training with DGL built-in GraphConv module.")

    # load and precess dataset
    transform = (
        AddSelfLoop()
    ) # by default, it will first remove self-loops to prevent duplication
    if args.dataset == "texas":
         data = TexasDataset(transform=transform)
    elif args.dataset == "chameleon":
         data = ChameleonDataset(transform=transform)
    else:
         raise ValueError("Unknown dataset: {}".format(args.dataset))

    g = data[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    features = g.ndata["feat"]
    # create GCN model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GCN(in_size, 16, out_size).to(device)

    # model training
    acc_list = []
    print("Training...")
    #for Ex_time in range(10):
    eta = 0.0001
    m = 0.0001
    a = 0.0001
    b = 0.0001
    for split in range(10):
        data = get_data(args.dataset, split).to(device)
        masks = data.train_mask, data.val_mask, data.test_mask
        features = data.x
        labels = data.y
        train(g, features, labels, masks, model, eta, m , a ,b)
        # model testing

        print("Testing....")
        acc = evaluate(g, features, labels, masks[2], model)
        acc_list.append(acc)
        print("Test accuracy {:.4f}".format(acc))

    print(acc_list)
    acc_mean = np.mean(acc_list)
    acc_var = np.std(acc_list)
    print('dataset', args.dataset)
    print("Test accuracy mean {:.4f}".format(acc_mean*100))
    print("Test accuracy variation {:.4f}".format(acc_var*100))
    print("eta:{}, m:{},a:{},b:{}".format(eta, m, a, b))