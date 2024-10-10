import numpy as np
import torch

from config import args


def cos_top(feature, top_k, min_value):
    # compute cosine similarity
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)  # variance normalization
    adj = torch.mm(feature, feature.T)  # matrix multiplication
    adj = adj - torch.eye(adj.shape[0])
    # top K
    if top_k < 0:
        top_k = adj.shape[0]
    # args.min_value
    filter_value = 0
    indices_to_remove = (adj < torch.topk(adj, top_k)[0][..., -1, None]) | (adj < torch.full([adj.shape[0], 1], min_value))
    adj[indices_to_remove] = filter_value
    adj = torch.triu(adj)
    return adj.detach()

# Covid19
dataset='Covid19'
graphStru='cos'
root_path = './dataset/output'
X_pca = torch.load('{}/{}/file/X_pca.pt'.format(root_path, dataset))
X_bcr = torch.load('{}/{}/file/X_bcr.pt'.format(root_path, dataset))
adj_pca = cos_top(X_pca, args.top_k, args.min_value)
adj_bcr = cos_top(X_bcr, args.top_k, args.min_value)
torch.save(adj_pca, '{}/{}/graph/{}/pca_cos_{}.pt'.format(root_path, dataset, graphStru, args.top_k))
torch.save(adj_bcr, '{}/{}/graph/{}/bcr_cos_{}.pt'.format(root_path, dataset, graphStru, args.top_k))