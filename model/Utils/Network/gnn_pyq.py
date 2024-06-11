#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
This file defines the Graph Neural Network (GNN).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import time
import sys
import numpy as np
from torch_sparse import SparseTensor

from logging import Logger
from collections import defaultdict
from torch_geometric.loader import DataLoader
from Msdroid.Network.dataset import MsdroidDataset
from Utils.Network.helper import get_device, eval_metrics, adjust_learning_rate
from Utils.Network.gcn_conv import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, num_layers:int,
                 in_channels:int,
                 hidden_channels:int,
                 out_channels:int,
                 dropout_ratio:float=0.2, 
                 gnn_type:str='gcn',
                 JK:str='last'):
        super().__init__()
        
        # GNN parameters
        self.num_layers = num_layers
        self.JK = JK
        self.dropout_ratio = dropout_ratio

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # message passing layers
        if num_layers < 1:
            raise ValueError('Number of GNN layers must be greater than 0.')
        for layer in range(num_layers):
            if layer == 0:
                self.convs.append(self.gnn_layer(gnn_type, in_channels, hidden_channels))
            else:
                self.convs.append(self.gnn_layer(gnn_type, hidden_channels, hidden_channels))
            self.batch_norms.append(pyg_nn.BatchNorm(hidden_channels))

        # graph classification layer (input graph representation: global_mean_pool + global_max_pool)
        graph_channels = hidden_channels * (num_layers + 1) if JK == 'concat' else hidden_channels
        self.graph_pred_linear = nn.Sequential(
            nn.Linear(graph_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_channels, out_channels)
        )

    def gnn_layer(self, gnn_type:str, in_channels:int, out_channels:int):
        """ Obtain GNN convolution layer """
        if gnn_type == 'gcn':
            return GCNConv(in_channels, out_channels)
        elif gnn_type == 'gat':
            return pyg_nn.GATConv(in_channels, out_channels)
        elif gnn_type == 'sage':
            return pyg_nn.SAGEConv(in_channels, out_channels)
        elif gnn_type == 'gin':
            return pyg_nn.GINConv(
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    nn.ReLU(), 
                    nn.Linear(out_channels, out_channels)
                )
            )
        else:
            raise ValueError('GNN type must be one of gcn, gin, gat, or sage.')

    def get_node_rep(self, x, edge_index):
        h_list = [x]
        for layer in range(self.num_layers):
            sys.stdout.flush()
            x = self.convs[layer](h_list[layer], edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout_ratio, training=self.training)
            if layer != self.num_layers - 1:
                x = self.batch_norms[layer](x)
            h_list.append(x)

        # Jumping Knowledge
        if self.JK == "last":
            node_x = h_list[-1]
        elif self.JK == "sum":
            node_x = torch.sum(torch.stack(h_list, dim=0), dim=0)
        elif self.JK == "concat":
            node_x = torch.cat(h_list, dim=1)
        else:
            raise NotImplementedError

        return node_x        

    def forward(self, x, edge_index, batch):
        # node representation
        node_x = self.get_node_rep(x, edge_index)

        # graph representation
        graph_x = torch.cat((pyg_nn.global_mean_pool(node_x, batch), pyg_nn.global_max_pool(node_x, batch)), dim=1)

        # graph classification
        y_pred = self.graph_pred_linear(graph_x)
        y_pred = F.log_softmax(y_pred, dim=1)
        return y_pred
    
    def get_graph_rep(self, x, edge_index, batch):
        # node representation
        node_x = self.get_node_rep(x, edge_index)

        # graph representation
        graph_x = torch.cat((pyg_nn.global_mean_pool(node_x, batch), pyg_nn.global_max_pool(node_x, batch)), dim=1)
        return graph_x
    
    def get_graph_pred(self, graph_x):
        # graph classification
        y_pred = self.graph_pred_linear(graph_x)
        y_pred = F.log_softmax(y_pred, dim=1)
        return y_pred

def apk_loss(pred, label, position):
    """ apk loss """
    loss = 0
    for i in range(len(position) - 1):
        start, end = position[i:i+2]
        apk_pred = pred[start:end]
        apk_label = label[start:end]

        unilabel = set(apk_label.tolist())
        assert len(unilabel) == 1
        unilabel = list(unilabel)[0]

        # benign apk
        if unilabel == 0:
            apk_loss = F.nll_loss(apk_pred, apk_label)
        else:
            # malware apk
            scores = []
            for j in range(end-start):
                scores.append(F.nll_loss(apk_pred[j:j+1], apk_label[j:j+1]))
            apk_loss = min(scores)
        
        loss += apk_loss
    
    return loss

def convert_real_batch(batch):
    """ convert batch to real batch 
    Model would be generated for APIs using APK labels.
    Batch Trick: 
        Input Batch is generated for APKs because we don't want to seperate the APIs inside. So the real batch size is not fixed for each. `position` indicates boundaries for each APK inside the batch.
    """
    sub_graphs = []
    position = [0]
    count = 0
    # import IPython; IPython.embed()
    for apk in batch.data:
        # Todo: skip the apk with too many apis
        if len(apk) > 400:
            continue
        for api in apk:
            sub_graphs.append(api)
        count += len(apk)
        position.append(count)
    
    sub_graphs = DataLoader(sub_graphs, batch_size=len(sub_graphs))
    
    # convert dataloader to batch
    for b in sub_graphs:
        """ one batch (batch_size=len(real))
        real_batch_size approximately equal to batch_size*avg(apk_subgraph_num) 
        """
        sub_graphs_batch = b
    
    return sub_graphs_batch, position

def msdroid_train(model_path: str, train_data: MsdroidDataset, val_data: MsdroidDataset, test_data: MsdroidDataset, model: GNN, logger: Logger,**kwargs):
    """
    Train GNN model for msdroid (the training process is not general)
    :param model_path: path to save trained model
    :param train_data: training dataset
    :param val_data: validation dataset
    :param test_data: testing dataset
    :param model: GNN model
    :param logger: logger
    :param kwargs: other arguments
    """
    device_id = kwargs['device']
    device = get_device(device_id)
    model = model.to(device)

    epochs = kwargs['epochs']
    lr = kwargs['lr']
    weight_decay = kwargs['weight_decay']
    batch_size = kwargs['batch_size']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_data.get_dataset(), batch_size=batch_size, shuffle=True)

    ret_val = defaultdict(lambda: 0)
    ret_test = defaultdict(lambda: 0)
    ret_val['f1'] = 0.

    # Early stopping
    patience = 16
    tolerance = 0.001
    no_improvement_count = 0
    # Total time
    total_train_time = 0.
    total_val_time = 0.

    for epoch in range(epochs):
        model.train()
        # adjust_learning_rate(optimizer, epoch, lr)
        total_loss = 0.

        time_start = time.time()
        for i, data in enumerate(train_loader):
            data, position = convert_real_batch(data)
            data.to(device)

            # training with adj matrix: must perform transpose
            # adj = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=2 * data.x.shape[:1]).t()
            # pred = model(data.x, adj, data.batch)

            # training with edge index
            pred = model(data.x, data.edge_index, data.batch)

            loss = apk_loss(pred, data.y, position)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Empty cache
            del data
            torch.cuda.empty_cache()

        time_end = time.time()
        train_time = time_end - time_start

        mean_loss = total_loss / len(train_loader)
        logger.info('Epoch %d, train loss: %.4f, time: %.2fs' % (epoch, mean_loss, train_time))
        total_train_time += train_time

        # Validation
        time_val_start = time.time()
        _, ret_val_tmp = msdroid_evaluate(val_data, model, **kwargs)
        time_val_end = time.time()
        val_time = time_val_end - time_val_start
        total_val_time += val_time
        logger.info(f'Validation F1: {ret_val_tmp["f1"]:.4f}, Recall: {ret_val_tmp["recall"]:.4f}, Precision: {ret_val_tmp["precision"]:.4f}, Accuracy: {ret_val_tmp["accuracy"]:.4f}, AUC: {ret_val_tmp["auc"]:.4f}, Time: {val_time:.2f}s')

        logger.warning(f'Total training time: {total_train_time:.2f}s, Total validation time: {total_val_time:.2f}s')

        # Save model
        if ret_val_tmp['f1'] > ret_val['f1'] + tolerance:
            ret_val = ret_val_tmp
            test_start_time = time.time()
            _, ret_test = msdroid_evaluate(test_data, model, **kwargs)
            test_end_time = time.time()
            torch.save(model, model_path)
            # Reset early stopping counter
            no_improvement_count = 0

            logger.debug(f'Test F1: {ret_test["f1"]:.4f}, Recall: {ret_test["recall"]:.4f}, Precision: {ret_test["precision"]:.4f}, Accuracy: {ret_test["accuracy"]:.4f}, AUC: {ret_test["auc"]:.4f}, Time: {test_end_time - test_start_time:.2f}s')
        else:
            no_improvement_count += 1
            # Early stopping
            if no_improvement_count >= patience:
                logger.warning(f'Early stopping at epoch {epoch}')
                break
    
    return ret_val, ret_test

def msdroid_evaluate(data: MsdroidDataset, model: GNN, **kwargs):
    """
    Evaluate GNN model for msdroid (the testing process is not general)
    :param data: testing dataset
    :param model: GNN model
    :param kwargs: other arguments
    """
    device_id = kwargs['device']
    device = get_device(device_id)

    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 1
    loader = DataLoader(data.get_dataset(), batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    
    api_preds = []
    apk_preds = []
    apk_labels = []

    for _, data in enumerate(loader):

        data, position = convert_real_batch(data)
        data.to(device)

        with torch.no_grad():
            data.to(device)

            # training with edge index
            output = model(data.x, data.edge_index, data.batch)

            pred = output.argmax(dim=1)
            label = data.y
            api_preds.append(pred)

            del data
            torch.cuda.empty_cache()
        
        for i in range(len(position) - 1):
            start, end = position[i:i+2]
            apk_pred = pred[start:end]
            apk_label = label[start:end]
            unilabel = set(apk_label.tolist())

            assert len(unilabel) == 1
            unilabel = list(unilabel)[0]
            apk_pred = apk_pred.sum().sign().item()
            apk_preds.append(apk_pred)
            apk_labels.append(unilabel)

    return api_preds, eval_metrics(apk_labels, apk_preds)
