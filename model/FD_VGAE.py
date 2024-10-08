#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
This file defines the FD_VAE.
"""
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import numpy as np
from tqdm import tqdm
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from model.base.abstract_run import AbstractRUN
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
from torch_geometric.nn.conv import GINConv, SAGEConv, GATConv, GCNConv
from model.base.utils import *
from logging import Logger
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
#from RAMDA.Network.dataset import RAMDADataset
from model.Utils.Network.helper import get_device, adjust_learning_rate, eval_metrics


class FD_VGAE_RUN(AbstractRUN):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.seed = args.seed
        self.lambda_1 = args.lambda1
        self.lambda_2 = args.lambda2
        self.lambda_3 = args.lambda3
        self.max_loss = np.inf

    def train_one_epoch(self, epoch):
        
        #train_data = self.data.construct_dataset(mode = "train")
        train_idx = list(range(self.data.n_train))

        train_loader = DataLoader(train_idx, batch_size=2*self.batch_size, shuffle=True, worker_init_fn=np.random.seed(self.seed))
        
        self.model.train()

        running_loss, running_marginal_likelihood, running_kl_divergence, running_disentangle_loss, num_batches = 0, 0, 0, 0, 0 

        pbar = tqdm(enumerate(train_loader), mininterval=2, total = len(train_loader))

        adjust_learning_rate(self.optimizer, epoch, self.lr)

        
        for batch_idx, data in pbar:

            if len(data) < 2*self.batch_size:
                continue

            data_0 = data[0:self.batch_size]
            data_1 = data[self.batch_size:]

            # access data 
            # one batch
            train_objects = self.data.construct_dataset(data_0, "train")
            data_hat = Batch.from_data_list(train_objects)
            data_hat.cuda(self.device)
            # second batch for comparison
            train_objects = self.data.construct_dataset(data_1, "train")
            data_pair = Batch.from_data_list(train_objects)
            data_pair.cuda(self.device)
            
            # batch of benign data
            train_objects = self.data.construct_dataset(data_0, "train_benign")
            data_benign = Batch.from_data_list(train_objects)
            data_benign.cuda(self.device)

            # Forward
            _, mu_i, logvar_i = self.model(data_hat.x, data_hat.edge_index)
            _, mu_j, logvar_j = self.model(data_pair.x, data_pair.edge_index)

            kl_i = self.model.get_kl_loss(mu_i, logvar_i)
            kl_j = self.model.get_kl_loss(mu_j, logvar_j)

            #z, mu, logvar = self.model(data_benign.x, data_benign.edge_index)

             # Calculate the loss
            #marginal_likelihood, kl_divergence = self.model.get_vae_loss(data_benign.x,  data_benign.edge_index, data_benign.batch)
            # Mean
            #marginal_likelihood = torch.mean(marginal_likelihood)
            kl_divergence = torch.mean(kl_i) + torch.mean(kl_j)

            disentangle_loss = get_disentangle_loss(mu_i, mu_j, data_hat.y, data_pair.y, data_hat.batch, data_pair.batch)
            
            #loss = -self.lambda_1 * marginal_likelihood + self.lambda_2 * kl_divergence + self.lambda_3 * disentangle_loss
            loss = self.lambda_2 * kl_divergence + self.lambda_3 * disentangle_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()  
           
            running_loss += loss.detach().item()
            #running_marginal_likelihood += (-self.lambda_1 * marginal_likelihood.detach().item())
            running_kl_divergence += (self.lambda_2 * kl_divergence.detach().item())
            running_disentangle_loss += (self.lambda_3 * disentangle_loss.detach().item())
            
            num_batches += 1

        return [running_loss/num_batches, running_kl_divergence/num_batches, running_disentangle_loss/num_batches]

        #return [running_loss/num_batches, running_marginal_likelihood/num_batches, \
        #        running_kl_divergence/num_batches, running_disentangle_loss/num_batches]
    
    def train(self) -> None:
        
        self.set_optimizer() # get the optimizer
        self.flag = False
        for epoch in range(self.start_epoch, self.max_epoch):
            # print(self.model.embed_user.weight)
            if self.flag: # early stop
                break
            # All models
            t1 = time.time()
            losses = self.train_one_epoch(epoch) # train one epoch
            t2 = time.time()
            self.document_running_loss(losses, epoch, t2-t1) # report the loss
            if (epoch + 1) % self.verbose == 0:
            #if (epoch + 1) % 30 == 0: # evaluate the model
                self.eval_and_check_early_stop(epoch, losses[0])     

    def eval_and_check_early_stop(self, epoch, loss):
        self.model.eval()

        if loss < self.max_loss:
            self.max_loss = loss
            save_checkpoint(self.model, epoch, self.base_path, self.checkpoint_buffer, self.args.max2keep)  
            self.data.best_valid_epoch = epoch
            self.data.patience = 0
        else:
            self.data.patience += 1
            if self.data.patience >= self.args.patience:
                print_str = "The best performance epoch is % d " % self.data.best_valid_epoch
                print(print_str)
                self.flag = True
                
    def execute(self):

        # write args
        perf_str = str(self.args)
        with open(self.base_path + 'stats.txt','a') as f:
            f.write(perf_str+"\n")

        self.model, self.start_epoch = self.restore_checkpoint(self.model, self.base_path, self.cuda) 

        if self.args.need_pretrain:
            print("start training") 
            self.train()
            self.model = self.restore_best_checkpoint(self.data.best_valid_epoch, self.model, self.base_path, self.cuda)
        
        print("start testing")
        self.model.eval() # evaluate the best model
        print_str = "The best epoch is % d" % self.data.best_valid_epoch
        with open(self.base_path +'stats.txt', 'a') as f:
            f.write(print_str + "\n")

        
        final_valid = self.evaluation(self.model, self.base_path, name = "valid")
        final_test = self.evaluation(self.model,self.base_path, name = "test")

    def evaluation(self, model, base_path, name = "valid"):

        if name == "train":
            evaluate_idx = list(range(self.data.n_train))
        elif name == "valid":
            evaluate_idx = list(range(self.data.n_valid))
        elif name == "test":
            evaluate_idx = list(range(self.data.n_test))
            
        eval_loader = DataLoader(evaluate_idx, batch_size=self.batch_size, shuffle=False)

        marginal_likelihoods = []

        true_labels = []
    
        with torch.no_grad():
            for batch in eval_loader:
                #batch = [x.cuda(self.device) for x in batch]
                evaluate_objects = self.data.construct_dataset(batch, name)
                batch = Batch.from_data_list(evaluate_objects)

                batch.cuda(self.device)

                marginal_likelihood, kl_divergence = self.model.get_vae_loss(self, batch.x,  batch.edge_index, batch.batch)
            
                # Calculate the loss
                marginal_likelihoods.append(-marginal_likelihood)
                true_labels.extend(batch.y.cpu().tolist())
        

        marginal_likelihoods = torch.cat(marginal_likelihoods, dim=0)
        predefined_loss = self.model.predefined_loss

        y_pred = (marginal_likelihoods > predefined_loss).float().cpu().numpy()
        
        f1 = f1_score(true_labels, y_pred)
        accuracy = accuracy_score(true_labels, y_pred)
        confusion = confusion_matrix(true_labels, y_pred)

        false_alarm_rate = confusion[0, 1] / (confusion[0, 0] + confusion[0, 1])
        miss_detection_rate = confusion[1, 0] / (confusion[1, 0] + confusion[1, 1])

        n_ret = {"false_alarm": false_alarm_rate, "miss_detection": miss_detection_rate, "f1_score": f1, "accuracy": accuracy}

        perf_str = name+':{}'.format(n_ret)
        print(perf_str)
        with open(base_path + 'stats.txt', 'a') as f:
            f.write(perf_str + "\n")

        return n_ret



class FD_VGAE(nn.Module):
    def __init__(self, args, in_channels: int = 492, hidden_channels: int = 64, out_channels: int = 16, dropout_ratio: float = 0.1, predefined_loss: float = 200):
        super(FD_VGAE, self).__init__()
        """
        Encoder and Decoder share same hidden channels
        Decoreder's input channels is equal to encoder's output channels
        Decoder's output channels is equal to encoder's input channels
        """
       
        self.predefined_loss = predefined_loss
        # GNN parameters
        self.device = torch.device(args.cuda)
        self.num_layers = args.num_layers
        #self.dropout_ratio = args.dropout_ratio
        #self.gnn_type = args.gnn_type
        #self.JK = args.JK
        self.batch_size = args.batch_size

        self.latent_dim = 32
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, self.latent_dim,  use_bn=False) 
        self.gcn_logvar = GCNConv(hidden_channels, self.latent_dim,  use_bn=False)      

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_encoder_output(self, x, edge_index):
        hidden = self.gcn1(x, edge_index)
        mu = self.gcn_mu(hidden, edge_index)
        logvar = self.gcn_logvar(hidden, edge_index)

        return mu, logvar
    
    def forward(self, x, edge_index):
        mu, logvar = self.get_encoder_output(x, edge_index)
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar
    
    def get_kl_loss(self, mu, logvar):
        
        kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return kl_divergence

    def get_vae_loss(self, x,  edge_index, batch):

        z, mu, logvar = self.forward(x, edge_index)
        adj_matrices = self.edge_index_to_batched_adj(edge_index, batch)

        recon_adj_list = []
        for graph_idx in batch.unique():
            graph_mask = batch == graph_idx
            z_graph = z[graph_mask]
            recon_adj_graph = torch.mm(z_graph, z_graph.t())
            recon_adj_list.append(recon_adj_graph)
        #recon_adj_batched = torch.stack(recon_adj_list, dim=0)

        losses = []

        for adj, recon_adj in zip(adj_matrices, recon_adj_list):
            assert adj.shape == recon_adj.shape, "Shapes of adjacency matrix and its reconstruction do not match"
            loss = F.binary_cross_entropy_with_logits(recon_adj, adj.to(self.device))
            losses.append(loss)

        marginal_likelihood  =  torch.stack(losses, dim=0).to(self.device)

        #marginal_likelihood = F.binary_cross_entropy_with_logits(x_batched_adj, recon_adj_batched)

        #marginal_likelihood = torch.mean(marginal_likelihood, dim = 1)

        kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return marginal_likelihood, kl_divergence       
    
    def edge_index_to_batched_adj(self, edge_index, batch):
        B = batch.max().item() + 1  # number of graphs in the batch
        adj_matrices = []

        for graph_idx in range(B):
            graph_mask = (batch[edge_index[0]] == graph_idx) & (batch[edge_index[1]] == graph_idx)
            edges_of_graph = edge_index[:, graph_mask]
            
            # Deduct min node index to make the smallest node index 0 for each graph
            edges_of_graph = edges_of_graph - edges_of_graph.min()

            num_nodes = edges_of_graph.max().item() + 1
            adj = torch.zeros(num_nodes, num_nodes)
            adj[edges_of_graph[0], edges_of_graph[1]] = 1
            adj_matrices.append(adj)

        return adj_matrices
    
    def predict(self, x,  edge_index, batch):
        
        mu, logvar, z = self.get_encoder_output(x, edge_index)
        adj_matrices = self.edge_index_to_batched_adj(edge_index, batch)

        recon_adj_list = []
        for graph_idx in batch.unique():
            graph_mask = batch == graph_idx
            z_graph = z[graph_mask]
            recon_adj_graph = torch.mm(z_graph, z_graph.t())
            recon_adj_list.append(recon_adj_graph)
        #recon_adj_batched = torch.stack(recon_adj_list, dim=0)

        losses = []

        for adj, recon_adj in zip(adj_matrices, recon_adj_list):
            assert adj.shape == recon_adj.shape, "Shapes of adjacency matrix and its reconstruction do not match"
            loss = F.binary_cross_entropy_with_logits(recon_adj, adj.to(self.device))
            losses.append(loss)

        marginal_likelihood  =  torch.stack(losses, dim=0).to(self.device)
        y_pred = marginal_likelihood > self.predefined_loss
        
        return y_pred


def get_disentangle_loss(mu_i, mu_j, y_i, y_j, batch_i, batch_j):
    # Disentagle loss
    # First determine whether i, j belong to the same class
    
    mu_readout_i = global_mean_pool(mu_i, batch_i)
    mu_readout_j = global_mean_pool(mu_j, batch_j)

    vector_y = torch.mean(torch.pow(y_j.float() - y_i.float(), 2))
    vector_mu = torch.mean(torch.pow(mu_readout_j - mu_readout_i, 2), dim=1)
    loss_bac = 60 * vector_y

    loss_0 = torch.mean(torch.multiply(vector_mu, 1 - vector_y))
    loss_1 = torch.mean(torch.multiply(torch.abs(F.relu(loss_bac-vector_mu)), vector_y))

    disentagle_loss = loss_0 + loss_1

    return disentagle_loss
