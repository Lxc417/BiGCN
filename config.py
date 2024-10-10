import argparse
import torch
import os

parser = argparse.ArgumentParser(description='embedding')

# 0. graph structure
parser.add_argument('--top_k', type=int, default=100, help='the value of top_k')
parser.add_argument('--min_value', type=int, default=0, help='min cos value')


# 1.dataset
# Covid19 NPC SLE
parser.add_argument('--dataset', type=str, default='Covid19', help='dataset')
parser.add_argument('--graphStru', type=str, default='cos', help='graph')
parser.add_argument('--graph_name', type=str, default='_cos_100', help='graph name')

parser.add_argument('--nfeat2', type=int, default=20, help='dim of input feature')
parser.add_argument('--nfeat1', type=int, default=100, help='dim of input feature')
parser.add_argument('--nfeat', type=int, default=120, help='dim of input feature')
parser.add_argument('--nhid', type=int, default=40, help='dim of hidden embedding')
parser.add_argument('--nout', type=int, default=40, help='dim of output embedding')
parser.add_argument('--save_data', type=int, default=1, help='if saving data')
# 2.experiments
parser.add_argument('--clip', type=int, default=5, help='clip of loss.')

parser.add_argument('--num_nodes', type=int, default=8723, help='number of node.')

parser.add_argument('--max_epoch', type=int, default=1500, help='number of epochs to train.')
parser.add_argument('--min_loss', type=int, default=1500, help='minimum loss.')
parser.add_argument('--device', type=str, default='cpu', help='training device')
parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')
parser.add_argument('--patience', type=int, default=50, help='patience for early stop')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight for L2 loss on basic models.')
parser.add_argument('--save_embeddings', type=int, default=1, help='save or not, default:1')
parser.add_argument('--sampling_times', type=int, default=1, help='negative sampling times')



# 3.models
parser.add_argument('--model', type=str, default='VGAE', help='model name')
parser.add_argument('--EPS', type=float, default=1e-15, help='eps')
parser.add_argument('--find_k', type=int, default=0, help='find the number of k')
parser.add_argument('--k', type=int, default=10, help='k clusters')
parser.add_argument('--max_k', type=int, default=100, help='k clusters')

args = parser.parse_args()

if args.dataset == 'Covid19':
    args.num_nodes = 8723
elif args.dataset == 'newFile':
    args.num_nodes = 1494