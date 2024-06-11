#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
This file defines the Multi-modal Neural Network (MNN).
"""
import torch
import torch.nn as nn
import time

from torch.utils.data import DataLoader, Dataset
from logging import Logger
from collections import defaultdict

from Utils.Network.helper import get_device, eval_metrics, adjust_learning_rate

class MNN(nn.Module):
    def __init__(self, modal_num: int, in_channel_list: list, init_hidden_layer_sizes: list, pred_hidden_layer_sizes: list, out_channels: int, dropout_ratio: float=0.2):
        super().__init__()
        self.modal_num = modal_num
        assert self.modal_num == len(in_channel_list)
        # Hard code the hidden layer sizes to be 3.
        assert len(init_hidden_layer_sizes) == 3

        self.modal_list = nn.ModuleList()
        for i in range(self.modal_num):
            self.modal_list.append(nn.Sequential(
                nn.Linear(in_channel_list[i], init_hidden_layer_sizes[0]),
                nn.ReLU(),
                nn.Dropout(dropout_ratio),
                nn.Linear(init_hidden_layer_sizes[0], init_hidden_layer_sizes[1]),
                nn.ReLU(),
                nn.Dropout(dropout_ratio),
                nn.Linear(init_hidden_layer_sizes[1], init_hidden_layer_sizes[2]),
                nn.ReLU()
            ))
        
        assert len(pred_hidden_layer_sizes) == 3
        self.pred_linear = nn.Sequential(
            nn.Linear(init_hidden_layer_sizes[2] * self.modal_num, pred_hidden_layer_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(pred_hidden_layer_sizes[0], pred_hidden_layer_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(pred_hidden_layer_sizes[1], pred_hidden_layer_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(pred_hidden_layer_sizes[2], out_channels)
        )

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
    
    def forward(self, x_list, y):
        assert len(x_list) == self.modal_num
        x_list = [self.modal_list[i](x_list[i]) for i in range(self.modal_num)]

        merged_x = torch.cat(x_list, dim=1)
        y_score = self.pred_linear(merged_x)

        y = y.view(-1, 1).float()
        loss = self.loss_fn(y_score, y)

        return loss
    
    def predict(self, x_list):
        assert len(x_list) == self.modal_num
        x_list = [self.modal_list[i](x_list[i]) for i in range(self.modal_num)]

        merged_x = torch.cat(x_list, dim=1)
        y_score = self.pred_linear(merged_x)
        # Sigmod to get the probability.
        y_prob = torch.sigmoid(y_score)
        # Get the predicted label.
        y_pred = torch.round(y_prob)

        return y_pred
    
def mnn_train(model_path: str, train_data: Dataset, val_data: Dataset, test_data: Dataset, model: MNN, logger: Logger, **kwargs):
    """
    Train the MNN model.
    :param model_path: The path to save the model.
    :param train_data: The training dataset.
    :param val_data: The validation dataset.
    :param test_data: The testing dataset.
    :param model: The MNN model.
    :param logger: The logger.
    :param kwargs: The other parameters.
    """
    device_id = kwargs['device_id']
    device = get_device(device_id)
    model = model.to(device)

    epochs = kwargs['epochs']
    lr = kwargs['lr']
    weight_decay = kwargs['weight_decay']
    batch_size = kwargs['batch_size']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    ret_val = defaultdict(lambda: 0)
    ret_test = defaultdict(lambda: 0)

    # Early stopping
    patience = 9
    tolerance = 0.001
    no_improvement_count = 0
    # Total training time
    total_train_time = 0.
    total_val_time = 0.

    for epoch in range(epochs):
        model.train()
        adjust_learning_rate(optimizer, epoch, lr)
        total_loss = 0.

        time_start = time.time()
        for x_list, y in train_loader:
            x_list = [x.to(device) for x in x_list]
            y = y.to(device)

            loss = model(x_list, y)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        time_end = time.time()
        train_time = time_end - time_start
        mean_loss = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch} | Train Loss: {mean_loss:.4f} | Time: {train_time:.2f}s')
        total_train_time += train_time

        time_val_start = time.time()
        ret_val_tmp = mnn_evaluate(val_data, model, **kwargs)
        time_val_end = time.time()
        val_time = time_val_end - time_val_start
        total_val_time += val_time
        logger.debug(f'Total training time: {total_train_time:.2f}s, Total validation time: {total_val_time:.2f}s')
        logger.info(f'Validation F1: {ret_val_tmp["f1"]:.4f}, Recall: {ret_val_tmp["recall"]:.4f}, Precision: {ret_val_tmp["precision"]:.4f}, Accuracy: {ret_val_tmp["accuracy"]:.4f}, AUC: {ret_val_tmp["auc"]:.4f}, Time: {val_time:.2f}s')
        if ret_val_tmp['f1'] > ret_val['f1'] + tolerance:
            ret_val = ret_val_tmp
            test_start_time = time.time()
            ret_test = mnn_evaluate(test_data, model, **kwargs)
            test_end_time = time.time()
            torch.save(model, model_path)
            # Reset the no improvement count
            no_improvement_count = 0

            logger.debug(f'Test F1: {ret_test["f1"]:.4f}, Recall: {ret_test["recall"]:.4f}, Precision: {ret_test["precision"]:.4f}, Accuracy: {ret_test["accuracy"]:.4f}, AUC: {ret_test["auc"]:.4f}, Time: {test_end_time - test_start_time:.2f}s')
        else:
            no_improvement_count += 1
            # Early stopping
            if no_improvement_count >= patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break
        
    return ret_val, ret_test

def mnn_evaluate(data: Dataset, model: MNN, **kwargs):
    """
    Evaluate the MNN model.
    :param data: The dataset.
    :param model: The MNN model.
    :param kwargs: The other parameters.
    """
    device_id = kwargs['device_id']
    device = get_device(device_id)
    batch_size = kwargs['batch_size']

    model = model.to(device)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x_list, y in loader:
            x_list = [x.to(device) for x in x_list]
            y = y.to(device)

            y_pred.append(model.predict(x_list))
            y_true.append(y)
    
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    y_true = torch.cat(y_true, dim=0).cpu().numpy()

    return eval_metrics(y_true, y_pred)