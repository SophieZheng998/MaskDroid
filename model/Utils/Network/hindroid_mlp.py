#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
This file defines the Multi-Layer Perceptron (MLP) for HinDroid (KDD'17).
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from logging import Logger
from collections import defaultdict

from Utils.Network.helper import get_device, eval_metrics

class MLP(torch.nn.Module):
    def __init__(self,
                 in_channels:int,
                 hidden_channels:int,
                 out_channels:int,
                 dropout_ratio:float=0.2,
                 matrix_A:np.array=None,
                 matrix_B:np.array=None,
                 matrix_P:np.array=None,
                 matrix_I:np.array=None):
        super().__init__()

        # alculate the kernels
        self.matrix_A = torch.from_numpy(matrix_A).to(get_device())
        self.matrix_B = torch.from_numpy(matrix_B).to(get_device())
        self.matrix_P = torch.from_numpy(matrix_P).to(get_device())
        self.matrix_I = torch.from_numpy(matrix_I).to(get_device())

        # Cannot be used if in_channels is huge (e.g., 125787 in Drebin). Otherwise, it causes GPU out of memory.
        self.att_mlp_layer = nn.Linear(in_channels, in_channels)

        self.pred_linear = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_channels, out_channels)
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        y_score = self.score(x)
        loss = self.loss_fn(y_score, y)
        return loss

    def predict(self, x):
        y_score = self.score(x)
        y_pred = torch.argmax(y_score, dim=1)
        return y_pred
    
    def score(self, x):
        kernel_0 = torch.mm(x, self.matrix_A.t())
        kernel_1 = torch.mm(torch.mm(x, self.matrix_B), self.matrix_A.t())
        kernel_2 = torch.mm(torch.mm(x, self.matrix_P), self.matrix_A.t())
        kernel_3 = torch.mm(torch.mm(x, self.matrix_I), self.matrix_A.t())
        kernel_4 = torch.mm(torch.mm(torch.mm(torch.mm(x, self.matrix_B), self.matrix_P), self.matrix_B.t()), self.matrix_A.t())
        kernel_5 = torch.mm(torch.mm(torch.mm(torch.mm(x, self.matrix_P), self.matrix_B), self.matrix_P.t()), self.matrix_A.t())
        kernel_6 = torch.mm(torch.mm(torch.mm(torch.mm(x, self.matrix_B), self.matrix_I), self.matrix_B.t()), self.matrix_A.t())
        kernel_7 = torch.mm(torch.mm(torch.mm(torch.mm(x, self.matrix_I), self.matrix_B), self.matrix_I.t()), self.matrix_A.t())
        kernel_8 = torch.mm(torch.mm(torch.mm(torch.mm(x, self.matrix_P), self.matrix_I), self.matrix_P.t()), self.matrix_A.t())
        kernel_9 = torch.mm(torch.mm(torch.mm(torch.mm(x, self.matrix_I), self.matrix_P), self.matrix_I.t()), self.matrix_A.t())
        kernel_10 = torch.mm(torch.mm(torch.mm(torch.mm(torch.mm(torch.mm(x, self.matrix_B), self.matrix_P), self.matrix_I), self.matrix_P.t()), self.matrix_B.t()), self.matrix_A.t())
        kernel_11 = torch.mm(torch.mm(torch.mm(torch.mm(torch.mm(torch.mm(x, self.matrix_P), self.matrix_B), self.matrix_I), self.matrix_B.t()), self.matrix_P.t()), self.matrix_A.t())
        kernel_12 = torch.mm(torch.mm(torch.mm(torch.mm(torch.mm(torch.mm(x, self.matrix_B), self.matrix_I), self.matrix_P), self.matrix_I.t()), self.matrix_B.t()), self.matrix_A.t())
        kernel_13 = torch.mm(torch.mm(torch.mm(torch.mm(torch.mm(torch.mm(x, self.matrix_I), self.matrix_B), self.matrix_P), self.matrix_B.t()), self.matrix_I.t()), self.matrix_A.t())
        kernel_14 = torch.mm(torch.mm(torch.mm(torch.mm(torch.mm(torch.mm(x, self.matrix_I), self.matrix_P), self.matrix_B), self.matrix_P.t()), self.matrix_I.t()), self.matrix_A.t())
        kernel_15 = torch.mm(torch.mm(torch.mm(torch.mm(torch.mm(torch.mm(x, self.matrix_P), self.matrix_I), self.matrix_B), self.matrix_I.t()), self.matrix_P.t()), self.matrix_A.t())

        kernel = torch.cat([kernel_0, kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6, kernel_7, kernel_8, kernel_9, kernel_10, kernel_11, kernel_12, kernel_13, kernel_14, kernel_15], dim=1)
        y_score = self.pred_linear(kernel)
        return y_score

def mlp_train(model:MLP, epochs:int, logger:Logger, train_data, val_data=None, test_data=None, model_path:str=None, device_id:int=0, lr:float=0.01, evaluation=True):
    # device to run the model
    device = get_device(device_id)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)

    ret_val = defaultdict(lambda: "Not present")
    ret_val['f1'] = 0.0
    ret_test = defaultdict(lambda: "Not present")
    for epoch in range(epochs):
        model.train()

        all_loss = 0.0
        for batch in train_loader:
            x = batch[0].float().to(device)
            y = batch[1].long().to(device)

            loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_loss += loss
        logger.info('Epoch: {}, Loss: {:.4f}'.format(epoch, all_loss))

        # we hardcode the validation frequency to 1
        if evaluation:
            ret_val_tmp = mlp_evaluate(val_data, model, device_id)
            if ret_val_tmp['f1'] > ret_val['f1']:
                ret_val = ret_val_tmp
                ret_test = mlp_evaluate(test_data, model, device_id)
                torch.save(model, model_path)
    
                logger.debug("Validation:\t[accuracy, recall, precision, f1]=[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(\
                            ret_val['accuracy'], ret_val['recall'], ret_val['precision'], ret_val['f1']))
                logger.debug("Testing:\t[accuracy, recall, precision, f1]=[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(\
                            ret_test['accuracy'], ret_test['recall'], ret_test['precision'], ret_test['f1']))

    return ret_val, ret_test

def mlp_evaluate(data, model:MLP, device_id:int=None):
    # device to run the model
    device = get_device(device_id)

    loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=4)

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
