import numpy
import torch

from config import args
from input_data import loader


def process(BCR_l, pca_l, adj_array):
    # graph --> index (x,y)
    # BCR --> X_BCR
    # exp --> BCR for search if same, average (pooling)
    # exp --> X_pce
    adj = torch.Tensor(adj_array)
    edge = torch.nonzero(adj).T

    X_bcr = []
    for i in range(len(BCR_l)):
        X_bcr.append(BCR_l[i].data)
    X_bcr = torch.tensor(numpy.array(X_bcr))

    X_pca = []
    for i in range(len(BCR_l)):
        count = 0
        tem = numpy.zeros(len(pca_l[0].data))
        for j in range(len(pca_l)):
            if BCR_l[i].BCR_id == pca_l[j].BCR_id:
                count = count + 1
                tem = tem + pca_l[j].data
        X_pca.append(tem / count)
    X_pca = torch.tensor(numpy.array(X_pca))
    return edge, X_bcr, X_pca


dataset = args.dataset

BCR_l, pca_l, adj_array = loader(dataset)

edge_index, X_bcr, X_pca = process(BCR_l, pca_l, adj_array)
if args.save_data:
    address = './dataset/output/{}/file'.format(dataset)
    torch.save(X_bcr, '{}/X_bcr.pt'.format(address))
    torch.save(X_pca, '{}/X_pca.pt'.format(address))
    torch.save(edge_index, '{}/edge_index.pt'.format(address))
    torch.save(adj_array, '{}/graph.pt'.format(address))
