#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
This file defines the Convolutional Neural Network (CNN).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from torch.utils.data import Dataset, DataLoader
from logging import Logger
from collections import defaultdict

from Utils.Network.helper import get_device, eval_metrics, adjust_learning_rate

class CNN(nn.Module):
    def __init__(self, in_channels: int, 
                 hidden_channels: int,
                 out_channels: int,
                 num_classes: int,
                 max_seq_len: int,
                 window_sizes: list, 
                 padding: str = 'same',
                 use_pooling: bool = True,
                 use_batch_norm: bool = False,
                 dropout_ratio: float = 0.2):
        super(CNN, self).__init__()
        self.use_pooling = use_pooling
        self.use_batch_norm = use_batch_norm
        self.dropout_ratio = dropout_ratio

        # Node embedding
        # self.node_embedding = nn.Linear(in_channels, hidden_channels)
        self.embedding = nn.Embedding(in_channels + 1, hidden_channels)

        # Convolutional layers
        self.convs = nn.ModuleList()
        # Pooling layers
        self.pools = nn.ModuleList()
        # Batch normalization layers
        self.bns = nn.ModuleList()
        for window_size in window_sizes:
            self.convs.append(nn.Conv1d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=window_size, padding=padding))
            if padding == 'same':
                self.pools.append(nn.MaxPool1d(kernel_size=max_seq_len))
            elif padding == 'valid':
                self.pools.append(nn.MaxPool1d(kernel_size=max_seq_len - window_size + 1))
            else:
                raise ValueError('Padding must be either "same" or "valid".')
            self.bns.append(nn.BatchNorm1d(out_channels))

        # Fully connected layer
        self.fc = nn.Linear(len(window_sizes) * out_channels, num_classes)
    
    def forward(self, x):
        # Node embedding
        # x = self.node_embedding(x)
        x = self.embedding(x)

        # Convolutional layers
        # batch_size, text_len, embedding_size -> batch_size, embedding_size, text_len
        x = x.permute(0, 2, 1)
        x = [conv(x) for conv in self.convs]
        x = [F.relu(x) for x in x]
        if self.use_pooling:
            x = [pool(x) for pool, x in zip(self.pools, x)]
        if self.use_batch_norm:
            x = [bn(x) for bn, x in zip(self.bns, x)]
        x = [F.dropout(x, p=self.dropout_ratio) for x in x]
        out = torch.cat(x, dim=1)

        # batch_size, embedding_size, text_len -> batch_size, embedding_size
        out = out.view(-1, out.size(1))

        # Fully connected layer
        out = F.relu(self.fc(out))

        return out

def cnn_train(model_path: str, train_data: Dataset, val_data: Dataset, test_data: Dataset, model: CNN, logger: Logger, **kwargs):
    """
    Train a CNN model.
    :param model_path: Path to save the trained model.
    :param train_data: Training dataset.
    :param val_data: Validation dataset.
    :param test_data: Test dataset.
    :param model: CNN model.
    :param logger: Logger.
    :param kwargs: Other parameters.
    """
    device_id = kwargs['device']
    device = get_device(device_id)
    model = model.to(device)

    epochs = kwargs['epochs']
    lr = kwargs['lr']
    weight_decay = kwargs['weight_decay']
    batch_size = kwargs['batch_size']
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    ret_val = defaultdict(lambda: 'Not present')
    ret_test = defaultdict(lambda: 'Not present')
    ret_val['f1'] = 0.

    # Early stopping
    patience = 9
    tolerance = 0.001
    no_improvement_count = 0
    # Total training time
    total_train_time = 0.
    taotal_val_time = 0.

    for epoch in range(epochs):
        model.train()
        adjust_learning_rate(optimizer, epoch, lr)
        total_loss = 0.

        time_start = time.time()
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        time_end = time.time()
        train_time = time_end - time_start
        mean_loss = total_loss / len(train_loader)
        logger.info('Epoch %d, mean loss: %.4f, time: %.2f' % (epoch, mean_loss, train_time))
        total_train_time += train_time
        time_val_start = time.time()
        ret_val_tmp = cnn_evaluate(val_data, model, **kwargs)
        time_val_end = time.time()
        val_time = time_val_end - time_val_start
        taotal_val_time += val_time
        logger.info(f'Validation F1: {ret_val_tmp["f1"]:.4f}, Recall: {ret_val_tmp["recall"]:.4f}, Precision: {ret_val_tmp["precision"]:.4f}, Accuracy: {ret_val_tmp["accuracy"]:.4f}, AUC: {ret_val_tmp["auc"]:.4f}, Time: {val_time:.2f}s')
        logger.warning(f'Total training time: {total_train_time:.2f}s, Total validation time: {taotal_val_time:.2f}s')
        if ret_val_tmp['f1'] > ret_val['f1'] + tolerance:
            ret_val = ret_val_tmp
            test_start_time = time.time()
            ret_test = cnn_evaluate(test_data, model, **kwargs)
            test_end_time = time.time()
            torch.save(model, model_path)
            # Reset early stopping counter
            no_improvement_count = 0

            logger.info(f'Test F1: {ret_test["f1"]:.4f}, Recall: {ret_test["recall"]:.4f}, Precision: {ret_test["precision"]:.4f}, Accuracy: {ret_test["accuracy"]:.4f}, AUC: {ret_test["auc"]:.4f}, Time: {test_end_time - test_start_time:.2f}s')
        else:
            no_improvement_count += 1
            # Early stopping
            if no_improvement_count >= patience:
                logger.warning(f'Early stopping at epoch {epoch}')
                break
        
    return ret_val, ret_test

def cnn_evaluate(data: Dataset, model: CNN, **kwargs):
    """
    Evaluate a CNN model.
    :param data: Dataset.
    :param model: CNN model.
    :param kwargs: Other parameters.
    """
    device_id = kwargs['device']
    batch_size = kwargs['batch_size']

    device = get_device(device_id)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    model.eval()

    y_true, y_pred = [], []
    for data, label in loader:
        data = data.to(device)
        label = label.to(device)

        out = model(data)
        pred = torch.argmax(out, dim=1)
        
        y_pred.append(pred)
        y_true.append(label)
    
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    y_true = torch.cat(y_true, dim=0).cpu().numpy()

    return eval_metrics(y_true, y_pred)

    