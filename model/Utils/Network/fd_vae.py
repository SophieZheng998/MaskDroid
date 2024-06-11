#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
This file defines the FD_VAE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

from logging import Logger
from RAMDA.Network.dataset import RAMDADataset
from Utils.Network.helper import get_device, adjust_learning_rate, eval_metrics

class FD_VAE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout_ratio: float = 0.1, predefined_loss: float = 20):
        super(FD_VAE, self).__init__()
        """
        Encoder and Decoder share same hidden channels
        Decoreder's input channels is equal to encoder's output channels
        Decoder's output channels is equal to encoder's input channels
        """

        self.predefined_loss = predefined_loss

        # Gaussian MLP Encoder
        self.gaussian_mlp_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_channels, out_channels * 2)
        )

        # Bernoulli MLP Decoder
        self.bernoulli_mlp_decoder = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.Tanh(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_channels, in_channels),
            nn.Sigmoid()
        )

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def reparameterise(self, mu, logvar):
        stddev = 1e-6 + F.softplus(logvar)
        # reparameterization trick
        eps = torch.randn_like(mu)

        z = mu + eps * stddev

        return mu, stddev, z
    
    def predict(self, x):
        mu, logvar = self.gaussian_mlp_encoder(x).chunk(2, dim=1)
        mu, logvar, z = self.reparameterise(mu, logvar)
        y = self.bernoulli_mlp_decoder(z)

        # clip the value of x_recon
        y = torch.clamp(y, 1e-8, 1 - 1e-8)

        # predict the label (negation)
        marginal_likelihood = -torch.sum(x * torch.log(y+1e-8) + (1 - x) * torch.log(1 - y + 1e-8), dim=1)
        y_pred = marginal_likelihood > self.predefined_loss
        return y_pred

    def forward(self, x):
        mu, logvar = self.gaussian_mlp_encoder(x).chunk(2, dim=1)
        mu, logvar, z = self.reparameterise(mu, logvar)
        y = self.bernoulli_mlp_decoder(z)

        # clip the value of x_recon
        y = torch.clamp(y, 1e-8, 1 - 1e-8)

        return y, mu, logvar
    
    def get_encoder_output(self, x):
        mu, logvar = self.gaussian_mlp_encoder(x).chunk(2, dim=1)
        mu, logvar, z = self.reparameterise(mu, logvar)

        return mu, logvar, z

def get_vae_loss(y, x, mu, logvar):
    # Reconstruction loss
    marginal_likelihood = torch.sum(x * torch.log(y+1e-8) + (1 - x) * torch.log(1 - y + 1e-8), dim=1)
    # marginal_likelihood = torch.mean(marginal_likelihood)

    # KL Divergence
    kl_divergence = 0.5 * torch.sum(mu**2 + logvar**2 - torch.log(1e-8 + logvar**2) - 1, dim=1)
    # kl_divergence = torch.mean(kl_divergence)

    return marginal_likelihood, kl_divergence

def get_disentagle_loss(mu_i, mu_j, y_i, y_j):
    # Disentagle loss
    # First determine whether i, j belong to the same class
    vector_y = torch.mean(torch.pow(y_j - y_i, 2))
    vector_mu = torch.mean(torch.pow(mu_j - mu_i, 2), dim=1)
    loss_bac = 60 * vector_y

    loss_0 = torch.mean(torch.multiply(vector_mu, 1 - vector_y))
    loss_1 = torch.mean(torch.multiply(torch.abs(F.relu(loss_bac-vector_mu)), vector_y))

    disentagle_loss = loss_0 + loss_1

    return disentagle_loss

def fd_vae_train(model_path, train_data: RAMDADataset, test_data: RAMDADataset, model: FD_VAE, logger: Logger, **kwargs):
    """
    Train the model.
    """
    device_id = kwargs['device_id']
    device = get_device(device_id)
    model.to(device)

    epochs = kwargs['epochs']
    batch_size = kwargs['batch_size']
    lr = kwargs['lr']
    lambda_1 = kwargs['lambda_1']
    lambda_2 = kwargs['lambda_2']
    lambda_3 = kwargs['lambda_3']

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_benign_data, train_malware_data = train_data.get_data()
    assert len(train_benign_data) == len(train_malware_data)

    # Batch number of each epoch
    sample_num = len(train_benign_data)
    total_batch = sample_num // batch_size

    # Training
    for epoch in range(epochs):
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, lr)
        total_loss = 0.
        total_marginal_likelihood = 0.
        total_kl_divergence = 0.
        total_disentagle_loss = 0.
        max_loss = np.inf

        # Training
        model.train()
        time_start = time.time()
        for batch_idx in range(total_batch):
            # Get the data
            offset = (batch_idx * batch_size) % sample_num
            batch_benign_data = train_benign_data[offset: offset + batch_size]
            batch_malware_data = train_malware_data[offset: offset + batch_size]
            batch_input = np.row_stack((batch_benign_data, batch_malware_data))
            np.random.shuffle(batch_input)

            batch_x_hat_data = batch_input[:, :-1]
            batch_x_hat_label = batch_input[:, -1]

            # Use next batch data to generate paired data for training
            offset = ((batch_idx + 1) * batch_size) % (sample_num - batch_size)
            batch_benign_data = train_benign_data[offset: offset + batch_size]
            batch_malware_data = train_malware_data[offset: offset + batch_size]
            batch_input = np.row_stack((batch_benign_data, batch_malware_data))
            np.random.shuffle(batch_input)

            batch_x_pair_data = batch_input[:, :-1]
            batch_x_pair_label = batch_input[:, -1]

            # Also select current batch benign data as x input for VAE
            batch_x_data = train_benign_data[offset: offset + batch_size]
            batch_x_data = np.row_stack((batch_x_data, batch_x_data))

            batch_x_data = batch_x_data[:, :-1]

            # Convert to tensor
            batch_x_hat_data = torch.from_numpy(batch_x_hat_data).float().to(device)
            batch_x_hat_label = torch.from_numpy(batch_x_hat_label).float().to(device)
            batch_x_pair_data = torch.from_numpy(batch_x_pair_data).float().to(device)
            batch_x_pair_label = torch.from_numpy(batch_x_pair_label).float().to(device)
            batch_x_data = torch.from_numpy(batch_x_data).float().to(device)

            # Forward
            mu_i, _, _ = model.get_encoder_output(batch_x_hat_data)
            mu_j, _, _ = model.get_encoder_output(batch_x_pair_data)
            y, mu, logvar = model(batch_x_data)

            # Calculate the loss
            marginal_likelihood, kl_divergence = get_vae_loss(y, batch_x_data, mu, logvar)
            # Mean
            marginal_likelihood = torch.mean(marginal_likelihood)
            kl_divergence = torch.mean(kl_divergence)

            disentagle_loss = get_disentagle_loss(mu_i, mu_j, batch_x_hat_label, batch_x_pair_label)
            
            loss = -lambda_1 * marginal_likelihood + lambda_2 * kl_divergence + lambda_3 * disentagle_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_marginal_likelihood += (-lambda_1 * marginal_likelihood.item())
            total_kl_divergence += (lambda_2 * kl_divergence.item())
            total_disentagle_loss += (lambda_3 * disentagle_loss.item())

        time_end = time.time()
        mean_loss = total_loss / total_batch
        mean_marginal_likelihood = total_marginal_likelihood / total_batch
        mean_kl_divergence = total_kl_divergence / total_batch
        mean_disentagle_loss = total_disentagle_loss / total_batch

        # Log the training process
        logger.info('Epoch: %d, Loss: %.4f, Marginal Likelihood: %.4f, KL Divergence: %.4f, Disentagle Loss: %.4f, Time: %.2f' % (epoch, mean_loss, mean_marginal_likelihood, mean_kl_divergence, mean_disentagle_loss, time_end - time_start))
        
        # Save the model
        if mean_loss < max_loss:
            max_loss = mean_loss
            torch.save(model, model_path)
        
        # Evaluate the model
        test_start_time = time.time()
        ret_test, y_pred, y_labels = fd_vae_evaluate(test_data, model, **kwargs)
        test_end_time = time.time()
        logger.debug("[FD-VAE] Testing:\t[accuracy, recall, precision, f1, time]=[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.2f}s]".format(ret_test['accuracy'], ret_test['recall'], ret_test['precision'], ret_test['f1'], test_end_time - test_start_time))
    
    return ret_test, y_pred, y_labels

def fd_vae_evaluate(test_dataset: RAMDADataset, model: FD_VAE, **kwargs):
    """
    Evaluate the model.
    """
    device_id = kwargs['device_id']
    device = get_device(device_id)
    model.to(device)

    batch_size = kwargs['batch_size']

    test_benign_data, test_malware_data = test_dataset.get_data()
    all_test_data = np.row_stack((test_benign_data, test_malware_data))

    test_data = all_test_data[:, :-1]
    test_label = all_test_data[:, -1]

    # Batch number of each epoch
    sample_num = len(test_data)
    # Batch number of each epoch
    
    total_batch = sample_num // batch_size
    if sample_num % batch_size:
        total_batch += 1

    # Testing
    model.eval()
    marginal_likelihoods = []

    with torch.no_grad():
        for batch_idx in range(total_batch):
            # Get the data
            start = (batch_idx * batch_size)
            if start + batch_size > sample_num:
                end = sample_num
            else:
                end = start + batch_size
            batch_data = test_data[start:end]

            # Convert to tensor
            batch_data = torch.from_numpy(batch_data).float().to(device)
            batch_data.to(device)
            # Forward
            y, mu, logvar = model(batch_data)

            marginal_likelihood, kl_divergence = get_vae_loss(y, batch_data, mu, logvar)
            assert marginal_likelihood.shape == (len(batch_data),)
            # Calculate the loss
            marginal_likelihoods.append(-marginal_likelihood)
    
    marginal_likelihoods = torch.cat(marginal_likelihoods, dim=0)
    predefined_loss = kwargs['predefined_loss']
    y_pred = (marginal_likelihoods > predefined_loss).float().cpu().numpy()
    ret = eval_metrics(test_label, y_pred)

    return ret, y_pred, test_label