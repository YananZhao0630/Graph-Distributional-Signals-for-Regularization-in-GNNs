# -*- coding:utf-8 -*-

import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
from utils import load_data
from model import CGNN
import argparse
from scipy import sparse
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import CoauthorCSDataset, AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset, CoauthorPhysicsDataset
from dgl import AddSelfLoop
from scipy import sparse as sp

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

def train(data, model, optimizer, g, eta, m, a, b):
    model.train()
    optimizer.zero_grad()
    pred = model(data)
    loss = F.nll_loss(pred[data.train_mask], data.y[data.train_mask])
    loss = loss + eta * LEReg_loss(g,pred, m, a,b)
    loss.backward()
    optimizer.step()


def val(data, model):
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    val_mask = data.val_mask
    accs.append(F.nll_loss(model(data)[val_mask], data.y[val_mask]))
    return accs


def main(args, d_input, d_output, g, eta, m, a,b):
    test_acc_list = []
    for i in range(args.num_expriment):
        data = load_data(args.data_path, args.dataset)
        data, model = globals()[args.model].call(data, args, d_input, d_output)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val_acc = test_acc = 0.0
        best_val_loss = np.inf
        wait_step = 0
        ##########################
        val_loss_list = []
        tem_test_acc_list = []
        for epoch in range(0, args.epochs):
            train(data, model, optimizer, g, eta, m,a,b)
            train_acc, val_acc, tmp_test_acc, val_loss = val(data, model)
            ##########################
            val_loss_list.append(val_loss.item())
            tem_test_acc_list.append(tmp_test_acc)
            if val_acc >= best_val_acc or val_loss <= best_val_loss:
                if val_acc >= best_val_acc:
                    test_acc = tmp_test_acc
                    early_val_acc = val_acc
                    early_val_loss = val_loss
                best_val_acc = np.max((val_acc, best_val_acc))
                best_val_loss = np.min((val_loss.cpu().detach().numpy(), best_val_loss))
                wait_step = 0
            else:
                wait_step += 1
                if wait_step == args.early_stop:
                    print('Early stop! Min loss: ', best_val_loss, ', Max accuracy: ', best_val_acc)
                    print('Early stop model validation loss: ', early_val_loss, ', accuracy: ', early_val_acc)
                    break
        log = 'Model_type: {}, Dateset_name: {}, Experiment: {:03d}, Test: {:.6f}'
        print(log.format(args.model_type, args.dataset, i + 1, test_acc))
        test_acc_list.append(test_acc * 100)
    log = 'Model_type: {}, Dateset_name: {}, Experiments: {:03d}, Mean: {:.6f}, std: {:.6f}, eta: {}, m:{}, a:{}, b:{}\n'
    print(log.format(args.model_type, args.dataset, args.num_expriment, np.mean(test_acc_list),
                     np.std(test_acc_list), eta, m, a, b))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CGNN')
    parser.add_argument('--data_path', type=str, help="Path of saved processed data files.", default='./data')
    parser.add_argument('--dataset', type=str, help="Name of the datasets", default='Cora') #,default='Cora'
    parser.add_argument('--NCTM', type=str, choices=['linear', 'exp'],
                        help="Type of Negative Curvature Transformation Module", default='linear')
    parser.add_argument('--CNM', type=str, choices=['symmetry-norm', '1-hop-norm', '2-hop-norm'],
                        help="Type of Curvature Normalization Module", default='symmetry-norm')
    parser.add_argument('--d_hidden', type=int, help="Dimension of the hidden node features", default=64)
    parser.add_argument('--epochs', type=int, help="The maximum iterations of training", default=200)
    parser.add_argument('--num_expriment', type=int, help="The number of the repeating expriments", default=50)
    parser.add_argument('--early_stop', type=int, help="Early stop", default=20)
    parser.add_argument('--dropout', type=float, help="Dropout", default=0.5)
    parser.add_argument('--lr', type=float, help="Learning rate", default=0.005)
    parser.add_argument('--weight_decay', type=float, help="Weight decay", default=0.0005)
    args = parser.parse_args()

    args.dataset = 'Cora'
    args.model = 'CGNN'
    datasets_config = {
        'Cora': {'d_input': 1433,
                 'd_output': 7},
        'Citeseer': {'d_input': 3703,
                     'd_output': 6},
        'PubMed': {'d_input': 500,
                   'd_output': 3},
        'CS': {'d_input': 6805,
               'd_output': 15},
        'Physics': {'d_input': 8415,
                    'd_output': 5},
        'Computers': {'d_input': 767,
                      'd_output': 10},
        'photo': {'d_input': 745,
                  'd_output': 8},
        'WikiCS': {'d_input': 300,
                   'd_output': 10},
    }

    args.model_type = 'CGNN_{}_{}_{}'.format(args.NCTM, args.CNM, args.dropout)
    d_input, d_output = datasets_config[args.dataset]['d_input'], datasets_config[args.dataset]['d_output']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset == 'Cora':
        transform = (AddSelfLoop())
        g = CoraGraphDataset(transform=transform)[0].int().to(device)
    if args.dataset == 'Citeseer':
        transform = (AddSelfLoop())
        g = CiteseerGraphDataset(transform=transform)[0].int().to(device)
    if args.dataset == "PubMed":
        transform = (AddSelfLoop())
        g = PubmedGraphDataset(transform=transform)[0].int().to(device)
    if args.dataset == 'CS':
        transform = (AddSelfLoop())
        g = CoauthorCSDataset(transform=transform)[0].int().to(device)
    if args.dataset == 'photo':
        transform = (AddSelfLoop())
        g = AmazonCoBuyPhotoDataset(transform=transform)[0].int().to(device)
    eta = 0.001
    m = 0.001
    a = 0.001
    b = 0.001
    main(args, d_input, d_output, g, eta, m, a,b)
