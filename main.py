# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time

import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt

from config import args
from input_data import loader, input_data, adj_to_sparse
from log import get_logger
from model import VGAENet, EMBFUSE


def main():
    # 1.input data
    global z, epochs
    X_pca, X_bcr, edge_index, adj_pca, adj_bcr = input_data(args.dataset, args.graphStru, args.graph_name)
    # 2.data precess    
    edge_index_pca, weight_pca = adj_to_sparse(edge_index, adj_pca)
    edge_index_bcr, weight_bcr = adj_to_sparse(edge_index, adj_bcr)

    # 3.set config, model and parameter
    patience = 0
    loss_l = []
    loss_pca_l = []
    loss_bcr_l = []
    min_loss = args.min_loss
    max_patience = args.patience
    model = EMBFUSE(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 4.train model
    logger = get_logger('./logger/{}.log'.format(time.strftime('%d_%H_%M', time.localtime())))
    logger.info('start training!')
    for epochs in tqdm(range(args.max_epoch)):
        optimizer.zero_grad()
        z, loss_pca, loss_bcr = model(edge_index_pca, weight_pca, edge_index_bcr, weight_bcr, X_pca.float(), X_bcr.float())
        loss = loss_pca + loss_bcr * 100
        loss_l.append(loss.detach().numpy())
        loss_pca_l.append(loss_pca.detach().numpy())
        loss_bcr_l.append(loss_bcr.detach().numpy())
        if loss.item() < min_loss:
            min_loss = loss.item()
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                print('early stopping')
                break
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
    # 5.save embedding
    if args.save_embeddings:
        address = './dataset/output/{}/embedding'.format(args.dataset)
        torch.save(z, "{}/embedding{}_balance.pt".format(address,args.graph_name))
        print('Embedding saved successfully!')
    # 6.show result
    print(np.array(loss_bcr_l).flatten())
    fig = plt.figure()
    f_loss_pca = fig.add_subplot(131)
    f_loss_pca.plot(range(epochs + 1), loss_pca_l)
    f_loss_bcr = fig.add_subplot(132)
    f_loss_bcr.plot(range(epochs + 1), loss_bcr_l)
    f_loss = fig.add_subplot(133)
    f_loss.plot(range(epochs+1), loss_l)
    plt.savefig("./dataset/output/{}/img/loss_{}_balance.png".format(args.dataset,args.graph_name))
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
