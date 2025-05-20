import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl import AddSelfLoop
import scipy
from scipy import sparse
from scipy.io import savemat
import argparse
import numpy as np
import time
#### GCN(base model + R-loss function that our proposed)
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

def new_loss_function(g,x):
    device = g.device
    prob = F.softmax(x,dim=1)
    num_nodes = g.number_of_nodes()

    adj = g.adj_external(scipy_fmt='csr').astype(float)
    # adj = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    Laplacian = sparse.eye(num_nodes) - adj
    Laplacian = torch.from_numpy(Laplacian.toarray()).float().to(device)

    y = torch.matmul(torch.matmul(torch.transpose(prob, 0, 1), Laplacian), prob)
    y = torch.trace(y)
    return y/num_nodes

def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0] # Extacts the train and validation masks from the 'masks' list
    val_mask = masks[1] # masks seperate the data into training and validation subsets
    loss_fcn = nn.CrossEntropyLoss() # defines the loss function as cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    # Initializes the Adam optimizer with the model parameters as the optimization variables.
    # training loop
    for epoch in range(200):
        start_time = time.time()
        model.train()  # set the model in the training mode
        logits = model(g, features)  # computes the logits (raw model outputs)
        eta = 0.001
        loss1 = loss_fcn(logits[train_mask], labels[train_mask])
        loss2 = new_loss_function(g,logits)
        loss = loss1+ eta*loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # End the timer
        end_time = time.time()

        # Calculate and print the total execution time
        total_time = end_time - start_time
        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}| time {:.4f}".format(
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
    # args.dataset = 'cora'
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
    # end_time = time.time()
    #
    # # Calculate and print the total execution time
    # total_time = end_time - start_time
    # print(f"Total running time: {total_time:.4f} seconds")





