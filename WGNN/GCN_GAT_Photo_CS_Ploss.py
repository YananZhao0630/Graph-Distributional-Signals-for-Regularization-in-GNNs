import torch
import numpy
import numpy as np
import scipy
import torch.nn as nn
import torch.nn.functional as F
import itertools
import scipy.sparse as sp
import statistics
import argparse
import time
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import CoauthorCSDataset, AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset, CoauthorPhysicsDataset
from base_models import *
from utils import *
import random
import statistics
import pickle
from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected
from scipy import sparse

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
    normalize_adj = torch.matmul(torch.linalg.pinv(deg_mat), adj)
    #normalize_adj = torch.matmul(torch.inverse(deg_mat), adj)
    # normalize_adj = torch.matmul(np.power(deg_mat,-1), adj)
    y = torch.matmul(normalize_adj, x.float())
    z = (torch.linalg.matrix_norm(y-x.float()))**2
    return z/num_nodes

def main(args, g_new=None, run=0, new_labels=None, new_train_mask=None):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    elif args.dataset == 'photo':
        data = AmazonCoBuyPhotoDataset()
    elif args.dataset == 'computer':
        data = AmazonCoBuyComputerDataset()
    elif args.dataset == 'physics':
        data = CoauthorPhysicsDataset()
    elif args.dataset == 'cs':
        data = CoauthorCSDataset()
    elif args.dataset in ['wisconsin', 'texas', 'cornell','chameleon']:
        data = get_data(args.dataset, split=run)
        if args.gpu < 0:
            features = data['x']
            labels = data['y']
        else:
            features = data['x'].to(args.gpu)
            labels = data['y'].to(args.gpu)
        data.edge_index = to_undirected(data.edge_index)
        adj = to_scipy_sparse_matrix(data.edge_index)
        g = dgl.from_scipy(adj)
        n_classes = int(data['y'].max() + 1)
        in_feats = features.size()[1]
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

    if args.dataset in ['cora','citeseer','pubmed','photo', 'computer', 'cs', 'physics']:
        g_original = g = data[0]
    else:
        g_original = g.clone()
    # g_original = dgl.remove_self_loop(g_original)

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    if args.dataset in ['citeseer', 'cora', 'pubmed']:
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_labels
    elif args.dataset in ['photo', 'computer', 'cs', 'physics']:
        features = g.ndata['feat']
        labels = g.ndata['label']
        in_feats = features.shape[1]
        n_classes = data.num_classes
        train_mask = args.train_mask
        val_mask = args.val_mask
        test_mask = args.test_mask

    # REPLACE TO USE G' HERE
    if g_new is not None:
      g = g_new
      if args.gpu < 0:
          cuda = False
      else:
          cuda = True
          g = g.int().to(args.gpu)

    n_edges = g.number_of_edges()

    print("""----Data statistics------'
      #Nodes %d
      #Edges %d""" % (g.number_of_nodes(), n_edges))

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    g = normalisation(g, cuda)

    if args.model == 'gcn':
        # create GCN model
        model = GCN(g, in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout)
    elif args.model == 'gat':
        # create GAT model
        heads = ([args.num_heads] * args.n_layers) + [args.num_out_heads]
        model = GAT(g, args.n_layers, in_feats, args.n_hidden, n_classes, heads, F.elu, args.in_drop, args.attn_drop, args.negative_slope, args.residual)
    elif args.model == 'mlp':
        model = MLP(in_feats, args.n_hidden, n_classes, F.relu, args.dropout)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.early_stop:
      patience = 100
      counter = 0
      best_score = 0

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)

        if new_train_mask is None:
            loss1 = loss_fcn(logits[train_mask], labels[train_mask])
        else:
            loss1 = loss_fcn(logits[new_train_mask], new_labels[new_train_mask])

        optimizer.zero_grad()
        eta_loss = args.eta_loss
        loss = loss1 + eta_loss * p_loss(g, logits)
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        if new_train_mask is None:
            acc, _ = evaluate(model, features, labels, val_mask)
        else:
            acc, _ = evaluate(model, features, new_labels, val_mask)

        if args.early_stop and epoch>100:
          if acc > best_score:
            best_score = acc
            counter = 0
          else:
            counter += 1
            if counter >= patience:
              break

        if epoch % 100 == 0:
          print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    print()
    # the testing step remains the same where the original labels and test_mask are used.
    acc, logits_test = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))

    return g_original, acc, logits_test, test_mask, train_mask, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--model", type=str, default="gcn",
                        help="model name ('gcn', 'gat').")
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed', 'wisconsin', 'texas', 'cornell','chameleon').")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument("--seed", type=int, default=0,
                        help="set seed")
    parser.add_argument("--save", action="store_true", default=False,
                        help="save g prime's edges? default false")
    parser.add_argument("--eta1", type=int, default=0,
                        help="set eta1")
    parser.add_argument("--eta2", type=int, default=0,
                        help="set eta2")
    parser.add_argument("--eta_loss", type=float, default=0.01,
                        help="hyperparameter for loss function")
    parser.add_argument("--all-combination", action="store_true", default=False,
                        help="run all eta1 and eta2 combinations?")
    parser.add_argument("--early-stop", action="store_true", default=False,
                        help="use early-stopping?")
    parser.add_argument("--step", type=int, default=10,
                        help="set step size for eta1 and eta2 for running all combinations")
    args = parser.parse_args()
    #args.dataset = 'photo'
    #args.model = 'gat'
    set_seed(args)
    if args.dataset in ['photo', 'computer', 'cs', 'physics']:
        set_masks(args)
    ############################################################
    # FIRSTLY : RUN THE PLAIN MODEL CAUSE WE NEED THE LOGITS #
    ############################################################
    # The plain model also coincides with the case where eta_1 = 0 and eta_2 = 0

    test_scores = []
    logits_test_dict = {}
    for run in range(10):
        g_original, test_acc, logits_test, test_mask, train_mask, labels = main(args, run=run)
        test_scores.append(test_acc)
        logits_test_dict[run] = logits_test
    #g_original = dgl.remove_self_loop(g_original)
    plain_score = (sum(test_scores)/len(test_scores)*100, statistics.stdev(test_scores)*100)
    print('plain model, test_acc ', args.model, plain_score)


