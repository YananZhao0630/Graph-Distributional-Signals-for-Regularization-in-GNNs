from data_handling import get_data
import numpy as np
import torch.optim as optim
from models import *
from torch import nn
from best_params import best_params_dict
from scipy import sparse as sp
from dgl.data import TexasDataset, ChameleonDataset
import argparse
from dgl import AddSelfLoop


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



def train_GNN(opt,split, eta, g, m, a,b):
    eta = eta
    g = g
    data = get_data(opt['dataset'],split)
    best_eval = 10000
    bad_counter = 0
    best_test_acc = 0

    if opt['model'] == 'GraphCON_GCN':
        model = GraphCON_GCN(nfeat=data.num_features,nhid=opt['nhid'],nclass=5,
                             dropout=opt['drop'],nlayers=opt['nlayers'],dt=1.,
                             alpha=opt['alpha'],gamma=opt['gamma'],res_version=opt['res_version']).to(opt['device'])
    elif opt['model'] == 'GraphCON_GAT':
        model = GraphCON_GAT(nfeat=data.num_features, nhid=opt['nhid'], nclass=5,
                             dropout=opt['drop'], nlayers=opt['nlayers'], dt=1.,
                             alpha=opt['alpha'], gamma=opt['gamma'],nheads=opt['nheads']).to(opt['device'])

    optimizer = optim.Adam(model.parameters(),lr=opt['lr'],weight_decay=opt['weight_decay'])
    lf = nn.CrossEntropyLoss()

    @torch.no_grad()
    def test(model, data):
        model.eval()
        logits, accs, losses = model(data), [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            loss = lf(out[mask], data.y.squeeze()[mask])
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
            losses.append(loss.item())
        return accs, losses

    for epoch in range(opt['epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(data.to(opt['device']))
        loss1 = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
        loss = loss1 + eta * LEReg_loss(g, out, m, a,b)
        loss.backward()
        optimizer.step()

        [train_acc, val_acc, test_acc], [train_loss, val_loss, test_loss] = test(model,data)

        if (val_loss < best_eval):
            best_eval = val_loss
            best_test_acc = test_acc
        else:
            bad_counter += 1

        if(bad_counter==opt['patience']):
            break

        log = 'Split: {:01d}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(split, epoch, train_acc, val_acc, test_acc))

    return best_test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--dataset', type=str, default='texas',
                        help='cornell, wisconsin, texas')
    parser.add_argument('--model', type=str, default='GraphCON_GCN',
                        help='GraphCON_GCN, GraphCON_GAT')
    parser.add_argument('--nhid', type=int, default=128,
                        help='number of hidden node features')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='alpha parameter of graphCON')
    parser.add_argument('--gamma', type=float, default=0,
                        help='gamma parameter of graphCON')
    parser.add_argument('--nheads', type=int, default=4,
                        help='number of attention heads for GraphCON-GAT')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='max epochs')
    parser.add_argument('--patience', type=int, default=100,
                        help='patience')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--drop', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--res_version', type=int, default=2,
                        help='version of residual connection')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight_decay')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='computing device')

    args = parser.parse_args()
    args.dataset = 'chameleon'
    cmd_opt = vars(args)

    # best_opt = best_params_dict[cmd_opt['dataset']]

    opt = {**cmd_opt}
    print(opt)

    n_splits = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eta = 0.0001
    # compute g
    transform = (AddSelfLoop())
    g = ChameleonDataset(transform=transform)[0]
    g = g.int().to(opt['device'])
    best = []
    m = 0.001
    a = 0.001
    b = 0.001
    for split in range(n_splits):
        best.append(train_GNN(opt,split, eta, g, m, a, b))
    print('Mean test accuracy: ', np.mean(np.array(best)*100),'std: ', np.std(np.array(best)*100), 'dataset:', args.dataset, 'eta:', eta, 'm:',m, 'a:', a, 'b:', b)

