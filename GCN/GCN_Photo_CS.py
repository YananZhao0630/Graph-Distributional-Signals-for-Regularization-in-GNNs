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
from data import get_dataset, set_train_val_test_split
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from dgl.data import CoauthorCSDataset, AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset, CoauthorPhysicsDataset
import time
## basic GCN module
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

def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0] # Extacts the train and validation masks from the 'masks' list
    val_mask = masks[1] # masks seperate the data into training and validation subsets
    loss_fcn = nn.CrossEntropyLoss() # defines the loss function as cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    # Initializes the Adam optimizer with the model parameters as the optimization variables.
    # training loop
    for epoch in range(500):
        model.train() # set the model in the training mode
        logits = model(g, features) # Computes the logits (raw model outputs)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
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
    start_time = time.time()
    parser = argparse.ArgumentParser()
    #parser.add_argument(
    #    "--dataset",
    #    type=str,
    #    default="cora",
    #    help="Dataset name ('cora', 'citeseer', 'pubmed').",
    #)
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")

    # load and precess dataset
    #transform = (
    #    AddSelfLoop()
    #) # by default, it will first remove self-loops to prevent duplication
    #if args.dataset == "cora":
    #     data = CoraGraphDataset(transform=transform)
    #elif args.dataset == "citeseer":
    #     data = CiteseerGraphDataset(transform=transform)
    #elif args.dataset == "pubmed":
    #     data = PubmedGraphDataset(transform=transform)
    #else:
    #    raise ValueError("Unknown dataset: {}".format(args.dataset))

    path = '../../data'
    ds = 'Photo'
    dataset = Amazon(path, ds)
    dataset.data = set_train_val_test_split(np.random.randint(0, 1000), dataset.data,
                                            num_development=5000 if ds == "CoauthorCS" else 1500)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = dataset.data.to(device)
    transform = (AddSelfLoop())
    g = AmazonCoBuyPhotoDataset(transform=transform)[0]
    #g = dgl.remove_self_loop(g)
    g = g.int().to(device)
    features = data.x
    labels = data.y
    masks = data.train_mask, data.val_mask, data.test_mask
    # create GCN model
    in_size = features.shape[1]
    out_size = dataset.num_classes
    model = GCN(in_size, 16, out_size).to(device)


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
    end_time = time.time()

    # Calculate and print the total execution time
    total_time = end_time - start_time
    print(f"Total running time: {total_time:.4f} seconds")


