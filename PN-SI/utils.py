import os, torch
import torch.nn.functional as F
from scipy import sparse

def train(net, optimizer, criterion, data, loss_type, eta, a, b, m):
    net.train()
    optimizer.zero_grad()
    output = net(data.x, data.adj)
    loss = criterion(output[data.train_mask], data.y[data.train_mask])
    if loss_type == 'R':
        prob = F.softmax(output, dim=1)
        # num_nodes = g.number_of_nodes()
        # adj = g.adj_external(scipy_fmt='csr').astype(float)
        # adj = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        # adj_scipy = sparse.csr_matrix(data.adj.cpu().numpy())
        num_nodes = data.adj.shape[0]
        # # Laplacian = sparse.eye(num_nodes) - adj_scipy
        Laplacian = torch.eye(num_nodes, device=data.adj.device) - data.adj
        # Laplacian = torch.from_numpy(Laplacian.toarray()).float()

        y = torch.matmul(torch.matmul(torch.transpose(prob, 0, 1), Laplacian), prob)
        y = torch.trace(y)
        R_loss = y/num_nodes
        loss = loss + eta*R_loss

    elif loss_type == 'P':
        num_nodes = data.adj.shape[0]
        adj = data.adj.to_dense()
        degrees = torch.sum(adj, dim=1)  # Sum along rows to get the degree of each node
        degree_matrix = torch.diag(degrees)
        y = torch.matmul(data.adj, output.float())
        z = (torch.linalg.matrix_norm(y - output.float())) ** 2
        p_loss = z/num_nodes
        loss = loss + eta*p_loss
    elif loss_type == 'L':
        P = F.softmax(output, dim=1)
        # X'LX
        XT = torch.transpose(output, 0, 1)  # cuda 0
        num_nodes = data.adj.shape[0]
        Laplacian = torch.eye(num_nodes, device=data.adj.device) - data.adj
        XTLX = torch.mm(torch.mm(XT, Laplacian), output)  # cuda 0 ---> L_intra = tr(XTLX)

        PT = torch.transpose(P, 0, 1)  # cuda 0
        B = torch.mm(torch.mm(PT, data.adj.to_dense().float()), P)
        degrees = B.sum(dim=1)  # Get degree (sum of each row)
        D_inv_sqrt = torch.diag(torch.pow(degrees, -0.5))  # Compute D^(-1/2)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0  # Handle division by zero
        B_normalized = torch.mm(torch.mm(D_inv_sqrt, B), D_inv_sqrt)
        L_B = torch.eye(B_normalized.shape[0]).to(B.device) - B_normalized
        # Y=P'X
        Y = torch.mm(PT, output)
        YT = torch.transpose(Y, 0, 1)
        # Y‘L_BY
        YTLBY = torch.mm(torch.mm(YT, L_B), Y)
        L_loss = a *torch.trace(XTLX)+ b* max(0, m-torch.trace(YTLBY))
        loss = loss + eta*L_loss
    else:
        loss = loss
    acc = accuracy(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, acc 

def val(net, criterion, data):
    net.eval()
    output = net(data.x, data.adj)
    loss_val = criterion(output[data.val_mask], data.y[data.val_mask])
    acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])
    return loss_val, acc_val

def test(net, criterion, data):
    net.eval()
    output = net(data.x, data.adj)
    loss_test = criterion(output[data.test_mask], data.y[data.test_mask])
    acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
    return loss_test, acc_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

