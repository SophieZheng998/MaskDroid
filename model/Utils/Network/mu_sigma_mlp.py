#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
This file defines the Mu-Sigma-Multi-Layer Perceptron (MU-SIGMA-MLP).
"""
import torch
import torch.nn as nn
import time
import numpy as np

from RAMDA.Network.dataset import RAMDADataset
from logging import Logger
from Utils.Network.helper import get_device, adjust_learning_rate, eval_metrics
from Utils.Network.fd_vae import FD_VAE

class MUSIGMA_MLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout_ratio: float = 0.1):
        super(MUSIGMA_MLP, self).__init__()

        self.mu_sigma_linear = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Tanh(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_channels, out_channels)
        )

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Loss function
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, mu_sigma, y):
        y_score = self.mu_sigma_linear(mu_sigma)
        y = torch.nn.functional.one_hot(y, num_classes=2).float()
        loss = self.loss_fn(y_score, y)

        return loss
    
    def predict(self, mu_sigma):
        y_score = self.mu_sigma_linear(mu_sigma)
        y_pred = torch.argmax(y_score, dim=1)
        return y_pred

def musigma_mlp_train(model_path, train_data: RAMDADataset, test_data: RAMDADataset, model: MUSIGMA_MLP, vae_model: FD_VAE, logger: Logger, **kwargs):
    """
    Train the model.
    """
    device_id = kwargs['device_id']
    device = get_device(device_id)
    model.to(device)
    vae_model.to(device)

    epochs = kwargs['epochs']
    batch_size = kwargs['batch_size']
    lr = kwargs['lr']

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_benign_data, train_malware_data = train_data.get_data()
    assert len(train_benign_data) == len(train_malware_data)

    sample_num = len(train_benign_data)
    total_batch = sample_num // batch_size

    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, lr)
        all_loss = 0.
        max_loss = np.inf

        model.train()
        time_start = time.time()
        for batch_idx in range(total_batch):
            # Get the data
            offset = (batch_idx * batch_size) % sample_num
            batch_benign_data = train_benign_data[offset:(offset + batch_size)]
            batch_malware_data = train_malware_data[offset:(offset + batch_size)]

            batch_input = np.row_stack((batch_benign_data, batch_malware_data))

            batch_x_data = batch_input[:, :-1]
            batch_x_label = batch_input[:, -1]

            batch_x_data = torch.from_numpy(batch_x_data).float()
            batch_x_label = torch.from_numpy(batch_x_label).long()

            batch_x_data, batch_x_label = batch_x_data.to(device), batch_x_label.to(device)

            # Train the model
            mu, sigma, z = vae_model.get_encoder_output(batch_x_data)

            # Concatenate mu and sigma
            mu_sigma = torch.cat((mu, sigma), dim=1)

            # Get the loss
            loss = model(mu_sigma, batch_x_label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_loss += loss.item()
        
        mean_loss = all_loss / total_batch
        time_end = time.time()
        logger.info('Epoch: %d, Loss: %.4f, Time: %.2f' % (epoch, mean_loss, time_end - time_start))

        # Save the model
        if mean_loss < max_loss:
            max_loss = mean_loss
            torch.save(model, model_path)
        
        # Evaluate the model
        test_start_time = time.time()
        ret_test, y_pred, y_labels = musigma_mlp_evaluate(test_data, model, vae_model, **kwargs)
        test_end_time = time.time()
        logger.debug("[MLP] Testing:\t[accuracy, recall, precision, f1, time]=[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.2f}s]".format(ret_test['accuracy'], ret_test['recall'], ret_test['precision'], ret_test['f1'], test_end_time - test_start_time))
    
    return ret_test, y_pred, y_labels

def musigma_mlp_evaluate(test_dataset: RAMDADataset, model: MUSIGMA_MLP, vae_model: FD_VAE, **kwargs):
    """
    Test the model.
    """
    device_id = kwargs['device_id']
    device = get_device(device_id)
    model.to(device)
    vae_model.to(device)

    batch_size = kwargs['batch_size']

    test_benign_data, test_malware_data = test_dataset.get_data()
    all_test_data = np.row_stack((test_benign_data, test_malware_data))
    test_data = all_test_data[:, :-1]
    test_label = all_test_data[:, -1]

    # Batch number
    sample_num = len(test_data)
    total_batch = sample_num // batch_size
    if sample_num % batch_size:
        total_batch += 1

    # Testing
    model.eval()
    y_pred = []

    with torch.no_grad():
        for batch_idx in range(total_batch):
            # Get the data
            start = batch_idx * batch_size
            if start + batch_size > sample_num:
                end = sample_num
            else:
                end = start + batch_size
            batch_data = test_data[start:end]

            # Convert to tesnsor
            batch_data = torch.from_numpy(batch_data).float().to(device)

            # Forward
            mu, sigma, z = vae_model.get_encoder_output(batch_data)
            mu_sigma = torch.cat((mu, sigma), dim=1)
            y_pred_x = model.predict(mu_sigma)
            y_pred.append(y_pred_x)
    
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()

    ret = eval_metrics(test_label, y_pred)

    return ret, y_pred, test_label