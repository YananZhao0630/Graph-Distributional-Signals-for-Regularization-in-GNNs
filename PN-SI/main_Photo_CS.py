import os, torch, logging, argparse
import models
from utils import train, test, val
from data import load_data
import numpy as np
# out dir
OUT_PATH = "results/"
if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)

# parser for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cora', help='{cora, pubmed, citeseer,photo}.')
parser.add_argument('--model', type=str, default='GCN', help='{SGC, DeepGCN, DeepGAT}')
parser.add_argument('--hid', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--nhead', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--log', type=str, default='debug', help='{info, debug}')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
# for deep model
parser.add_argument('--nlayer', type=int, default=2, help='Number of layers, works for Deep model.')
parser.add_argument('--residual', type=int, default=0, help='Residual connection')
# for PairNorm
# - PairNorm mode, use PN-SI or PN-SCS for GCN and GAT. With more than 5 layers get lots improvement.
parser.add_argument('--norm_mode', type=str, default='None', help='Mode for PairNorm, {None, PN, PN-SI, PN-SCS}')
parser.add_argument('--norm_scale', type=float, default=1.0, help='Row-normalization scale')
# for data
parser.add_argument('--no_fea_norm', action='store_false', default=True, help='not normalize feature' )
parser.add_argument('--missing_rate', type=int, default=0, help='missing rate, from 0 to 100' )
parser.add_argument('--loss_type', type=str, choices=['R', 'L', 'P'],help='Type of loss function')
parser.add_argument('--eta', type=float, default=0.1, help='loss regularization parameter')
parser.add_argument('--a', type=float, default=0.001, help='L-loss parameter a')
parser.add_argument('--b', type=float, default=0.001, help='L-loss parameter b')
parser.add_argument('--m', type=float, default=0.001, help='L-loss parameter m')
args = parser.parse_args()

# logger
#filename='example.log'
logging.basicConfig(format='%(message)s', level=getattr(logging, args.log.upper()))

# load data
data = load_data(args.data, normalize_feature=args.no_fea_norm, missing_rate=args.missing_rate, cuda=True)
nfeat = data.x.size(1)
nclass = int(data.y.max()) + 1
# adj = data.adj
# num_nodes = data.num_nodes
loss_type = args.loss_type
net = getattr(models, args.model)(nfeat, args.hid, nclass,
                                  dropout=args.dropout,
                                  nhead=args.nhead,
                                  nlayer=args.nlayer,
                                  norm_mode=args.norm_mode,
                                  norm_scale=args.norm_scale,
                                  residual=args.residual)
net = net.cuda()
optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.wd)
criterion = torch.nn.CrossEntropyLoss()
logging.info(net)

# train
best_acc = 0
best_loss = 1e10
test_acc_list = []

# Loop to test the model 10 times
for i in range(10):
    for epoch in range(args.epochs):
        train_loss, train_acc = train(net, optimizer, criterion, data, loss_type=loss_type, eta=args.eta, a=args.a, b=args.b, m=args.m)
        val_loss, val_acc = val(net, criterion, data)
        logging.debug('Epoch %d: train loss %.3f train acc: %.3f, val loss: %.3f val acc %.3f.'%
                    (epoch, train_loss, train_acc, val_loss, val_acc))
        # save model
        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), OUT_PATH+'checkpoint-best-acc.pkl')
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), OUT_PATH+'checkpoint-best-loss.pkl')

    # pick up the best model based on val_acc, then do test

    # Store test accuracy results in a list

        # Load the best model based on validation accuracy
    net.load_state_dict(torch.load(OUT_PATH + 'checkpoint-best-acc.pkl'))

    # Perform testing
    val_loss, val_acc = val(net, criterion, data)
    test_loss, test_acc = test(net, criterion, data)

        # Log results
    logging.info("-" * 50)
    logging.info("Run %d - Vali set results: loss %.3f, acc %.3f." % (i + 1, val_loss, val_acc))
    logging.info("Run %d - Test set results: loss %.3f, acc %.3f." % (i + 1, test_loss, test_acc))

    # Append the test accuracy to the list
    test_acc_list.append(test_acc.item())

    # Calculate the mean of the test accuracies
mean_test_acc = np.mean(test_acc_list)
std_test_acc = np.std(test_acc_list)
logging.info("=" * 50)
print("Mean test accuracy over 10 runs: %.4f." % (mean_test_acc*100))
print("Std test accuracy over 10 runs: %.4f." % (std_test_acc*100))


