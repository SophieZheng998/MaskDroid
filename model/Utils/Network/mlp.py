#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
This file defines the Multi-Layer Perceptron (MLP).
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader
from logging import Logger
from collections import defaultdict

from Utils.Network.helper import get_device, eval_metrics

class MLP(torch.nn.Module):
    def __init__(self,
                 in_channels:int,
                 hidden_channels:int,
                 out_channels:int,
                 attention:bool=False,
                 dropout_ratio:float=0.2):
        super().__init__()
        
        # Cannot be used if in_channels is huge (e.g., 125787 in Drebin). Otherwise, it causes GPU out of memory.
        self.attention = attention
        if self.attention:
            if in_channels > 2000:
                raise ValueError('Cannot use attention in MLP if in_channels is huge.')
            self.att_mlp_layer = nn.Linear(in_channels, in_channels)

        self.pred_linear = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_channels, out_channels)
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        if self.attention:
            QK = self.att_mlp_layer(x)
            QK = F.softmax(QK, dim=1)
            x = torch.mul(x, QK)
        y_score = self.pred_linear(x)
        loss = self.loss_fn(y_score, y)
        return loss

    def predict(self, x):
        if self.attention:
            QK = self.att_mlp_layer(x)
            QK = F.softmax(QK, dim=1)
            x = torch.mul(x, QK)
        y_score = self.pred_linear(x)
        y_pred = torch.argmax(y_score, dim=1)
        return y_pred
    
    def score(self, x):
        if self.attention:
            QK = self.att_mlp_layer(x)
            QK = F.softmax(QK, dim=1)
            x = torch.mul(x, QK)
        y_score = self.pred_linear(x)
        return y_score

def mlp_train(model:MLP, logger:Logger, train_data, val_data=None, test_data=None, model_path:str=None, evaluation=True, device = 0):
    # device to run the model
    device_id = device
    lr = 1e-3
    epochs = 10
    batch_size = 32
    device = get_device(device_id)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    ret_val = defaultdict(lambda: "Not present")
    ret_val['f1'] = 0.0
    ret_test = defaultdict(lambda: "Not present")

    # Early stopping
    patitence = 3
    tolerance = 0.001
    no_improvement_count = 0
    # Total time
    total_train_time = 0.
    total_val_time = 0.

    for epoch in range(epochs):
        model.train()
        train_start_time = time.time()
        all_loss = 0.0
        for batch in train_loader:
            x = batch[0].float().to(device)
            y = batch[1].long().to(device)

            loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_loss += loss
        train_end_time = time.time()
        train_time = train_end_time - train_start_time

        logger.info('Epoch: {}, Loss: {:.4f}, Time: {:.2f}s'.format(epoch, all_loss, train_end_time-train_start_time))
        total_train_time += train_time

        if evaluation:
            # Validation
            val_start_time = time.time()
            ret_val_tmp = mlp_evaluate(val_data, model, device_id, None)
            val_end_time = time.time()
            val_time = val_end_time - val_start_time
            total_val_time += val_time
            logger.debug("Validation:[f1, recall, precision, accuracy, time]=[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(ret_val_tmp['f1'], ret_val_tmp['accuracy'], ret_val_tmp['precision'], ret_val_tmp['f1'], val_time))

            logger.warning("Total training time: {:.2f}s, Total validation time: {:.2f}s".format(total_train_time, total_val_time))

            if ret_val_tmp['f1'] > ret_val['f1'] + tolerance:
                ret_val = ret_val_tmp
                test_start_time = time.time()
                ret_test = mlp_evaluate(test_data, model, device_id, None)
                test_end_time = time.time()
                torch.save(model, model_path)
                # Reset the no_improvement count
                no_improvement_count = 0

                logger.debug("Testing:[f1, recall, precision, accuracy, time]=[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.2}s]".format(ret_test['f1'], ret_test['recall'], ret_test['precision'], ret_test['accuracy'], test_end_time-test_start_time))
            else:
                no_improvement_count += 1
                # Early stopping
                if no_improvement_count >= patitence:
                    logger.warning("Early stopping at epoch {}".format(epoch))
                    break

    return ret_val, ret_test

def mlp_evaluate(data, model:MLP, device_id:int, test_files):
    # device to run the model
    device = get_device(device_id)

    loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=4)

    model.eval()
    y_pred, y_true = [], []
    for batch in loader:
        x = batch[0].to(device).float()
        y = batch[1].to(device).float()

        with torch.no_grad():
            y_pred_x = model.predict(x)     

        y_pred.append(y_pred_x)
        y_true.append(y)

    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    y_true = torch.cat(y_true, dim=0).cpu().numpy()

    ret = eval_metrics(y_true, y_pred)

    return ret
