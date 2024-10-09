# graph --> index (x,y)
# BCR --> Z_BCR
# exp --> BCR for search if same, average (pooling)
# exp --> Z_pce
import csv

import numpy
import numpy as np
import torch
import xlrd


def loader(dataset='Covid19'):
    class BCR:
        def __init__(self):
            self.BCR_id = ''
            self.BCR_name = ''
            self.data = []

    class pca:
        def __init__(self):
            self.pca_id = ''
            self.BCR_id = ''
            self.data = []

    adj_array = []
    BCR_l = []  
    pca_l = [] 
    if dataset == 'Covid19':
        root_path = './dataset/{}'.format(dataset)
        BCR_data = xlrd.open_workbook("{}/BCR_embedding.xlsx".format(root_path)).sheet_by_index(0)

        for i in range(BCR_data.nrows):
            row = BCR_data.row_values(i)
            tem = BCR()
            tem.BCR_id = row[0]
            tem.BCR_name = row[1]
            tem.data = row[2:]
            BCR_l.append(tem)
        exp_pca_data = xlrd.open_workbook("{}/exp_pca.xlsx".format(root_path)).sheet_by_index(0)

        for i in range(exp_pca_data.nrows):
            row = exp_pca_data.row_values(i)
            tem = pca()
            tem.pca_id = row[0]
            tem.BCR_id = row[1]
            tem.data = row[2:]
            pca_l.append(tem)
        # read crudegraph.csv
        with open("{}/crudegraph.csv".format(root_path), mode='r') as file:
            f_csv = csv.reader(file)
            for row in f_csv:
                adj_array.append(list(map(float, row)))
            adj_array = numpy.array(adj_array)
    elif dataset == 'newFile':
        root_path = './dataset/{}/'.format(dataset)
        with open("{}/BCR_embedding.csv".format(root_path), mode='r') as file:
            f_csv = csv.reader(file)
            for row in f_csv:
                tem = BCR()
                tem.BCR_id = row[1]
                tem.BCR_name = row[2]
                tem.data = list(map(float, row[3:]))
                BCR_l.append(tem)

        with open("{}/exp_pca.csv".format(root_path), mode='r') as file:
            f_csv = csv.reader(file)
            for row in f_csv:
                tem = pca()
                tem.pca_id = row[0]
                tem.BCR_id = row[1]
                tem.data = list(map(float, row[2:]))
                pca_l.append(tem)

        with open("{}/crudegraph.csv".format(root_path), mode='r') as file:
            f_csv = csv.reader(file)
            for row in f_csv:
                adj_array.append(list(map(float, row)))
            adj_array = numpy.array(adj_array)

    return BCR_l, pca_l, adj_array

def input_data(dataset='Covid19', graphStru='cos', graph_name='_cos_100'):
    root_path = './dataset/output'
    X_pca = torch.load('{}/{}/file/X_pca.pt'.format(root_path, dataset))
    X_bcr = torch.load('{}/{}/file/X_bcr.pt'.format(root_path, dataset))
    edge_index = torch.load('{}/{}/file/edge_index.pt'.format(root_path, dataset))
    adj_pca = torch.load('{}/{}/graph/{}/pca{}.pt'.format(root_path, dataset, graphStru, graph_name))
    adj_bcr = torch.load('{}/{}/graph/{}/bcr{}.pt'.format(root_path, dataset, graphStru, graph_name))

    return X_pca, X_bcr, edge_index, adj_pca, adj_bcr


def adj_to_sparse(edge_index, adj_tensor):
    adj_np = np.array(adj_tensor)
    edge_index = edge_index.numpy()
    edge_index = tuple(map(tuple, edge_index.tolist()))
    # print(type(edge_index))
    # edge_index = adj_np.nonzero()
    # print(type(edge_index))
    weight = adj_np[edge_index]

    edge_index = torch.tensor(edge_index)
    weight = torch.tensor(weight).float()

    return edge_index, weight

# loader()