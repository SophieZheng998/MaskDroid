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
#from model.Utils.Network.helper import adjust_learning_rate

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.5 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class FD_VAE_RUN(AbstractRUN):
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
            mu_i, _, _ = self.model.get_encoder_output(data_hat.x, data_hat.edge_index, data_hat.batch)
            mu_j, _, _ = self.model.get_encoder_output(data_pair.x, data_pair.edge_index, data_pair.batch)
            mu, logvar, y = self.model(data_benign.x, data_benign.edge_index, data_benign.batch)

             # Calculate the loss
            marginal_likelihood, kl_divergence = self.model.get_vae_loss(y, data_benign.x, data_benign.edge_index, data_benign.batch, mu, logvar)
            # Mean
            marginal_likelihood = torch.mean(marginal_likelihood)
            kl_divergence = torch.mean(kl_divergence)

            disentangle_loss = get_disentangle_loss(mu_i, mu_j, data_hat.y, data_pair.y)
            
            loss = -self.lambda_1 * marginal_likelihood + self.lambda_2 * kl_divergence + self.lambda_3 * disentangle_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()  
           
            running_loss += loss.detach().item()
            running_marginal_likelihood += (-self.lambda_1 * marginal_likelihood.detach().item())
            running_kl_divergence += (self.lambda_2 * kl_divergence.detach().item())
            running_disentangle_loss += (self.lambda_3 * disentangle_loss.detach().item())
            
            num_batches += 1

        return [running_loss/num_batches, running_marginal_likelihood/num_batches, \
                running_kl_divergence/num_batches, running_disentangle_loss/num_batches]
    
    # define the training process
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
                mu, logvar, y = model(batch.x, batch.edge_index, batch.batch)
                marginal_likelihood, kl_divergence = self.model.get_vae_loss(y, batch.x, batch.edge_index, batch.batch, mu, logvar)
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



class FD_VAE(nn.Module):
    def __init__(self, args, in_channels: int = 128, hidden_channels: int = 64, out_channels: int = 64, dropout_ratio: float = 0.1, predefined_loss: float = -150000):
        super(FD_VAE, self).__init__()
        """
        Encoder and Decoder share same hidden channels
        Decoreder's input channels is equal to encoder's output channels
        Decoder's output channels is equal to encoder's input channels
        """
       
        self.predefined_loss = predefined_loss
        # GNN parameters
        self.device = torch.device(args.cuda)
        self.num_layers = args.num_layers
        self.dropout_ratio = args.dropout_ratio
        self.gnn_type = args.gnn_type
        self.JK = args.JK
        self.batch_size = args.batch_size

        self.in_channels_original = args.in_channels
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.convs.append(self.gnn_layer(self.gnn_type, self.in_channels_original, in_channels))
        self.batch_norms.append(BatchNorm(in_channels))
       
        # Gaussian MLP Encoder
        # self.gaussian_mlp_encoder = nn.Sequential(
        #     nn.Linear(2 * in_channels, hidden_channels),
        #     nn.BatchNorm1d(hidden_channels),
        #     nn.ReLU(),
        #     # nn.Dropout(dropout_ratio),
        #     nn.Linear(hidden_channels, hidden_channels),
        #     nn.BatchNorm1d(hidden_channels),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_ratio),
        #     nn.Linear(hidden_channels, out_channels * 2)
        # )

        self.gaussian_mlp_encoder = nn.Sequential(
            nn.Linear(2 * in_channels, hidden_channels),
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
    
    def gnn_layer(self, gnn_type:str, in_channels:int, out_channels:int):
        """ Obtain GNN convolution layer """
        if gnn_type == 'gcn':
            return GCNConv(in_channels, out_channels)
        elif gnn_type == 'gat':
            return GATConv(in_channels, out_channels)
        elif gnn_type == 'sage':
            return SAGEConv(in_channels, out_channels)
        elif gnn_type == 'gin':
            return GINConv(
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
            x = F.dropout(x, self.dropout_ratio, training=True)
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

    def reparameterise(self, mu, logvar):
        stddev = 1e-6 + F.softplus(logvar)
        # reparameterization trick
        eps = torch.randn_like(mu)

        z = mu + eps * stddev

        return mu, stddev, z
    
    def predict(self, x,  edge_index, batch):
        # graph representation
        node_x = self.get_node_rep(x, edge_index)
        graph_x = torch.cat((global_mean_pool(node_x, batch), global_max_pool(node_x, batch)), dim=1)

        mu, logvar = self.gaussian_mlp_encoder(graph_x).chunk(2, dim=1)
        mu, logvar, z = self.reparameterise(mu, logvar)

        y = self.bernoulli_mlp_decoder(z)

        # clip the value of x_recon
        y = torch.clamp(y, 1e-8, 1 - 1e-8)

        # predict the label (negation)
        marginal_likelihood = -torch.sum(graph_x * torch.log(y+1e-8) + (1 - x) * torch.log(1 - y + 1e-8), dim=1)
        y_pred = marginal_likelihood > self.predefined_loss
        return y_pred

    def forward(self, x, edge_index, batch):

        # graph representation
        node_x = self.get_node_rep(x, edge_index)
        graph_x = torch.cat((global_mean_pool(node_x, batch), global_max_pool(node_x, batch)), dim=1)

        mu, logvar = self.gaussian_mlp_encoder(graph_x).chunk(2, dim=1)
        mu, logvar, z = self.reparameterise(mu, logvar)
       
        y = self.bernoulli_mlp_decoder(z)

        # clip the value of x_recon
        y = torch.clamp(y, 1e-8, 1 - 1e-8)

        return mu, logvar, y
        # return graph_x
    
    def get_encoder_output(self, x, edge_index, batch):

        # graph representation
        node_x = self.get_node_rep(x, edge_index)
        graph_x = torch.cat((global_mean_pool(node_x, batch), global_max_pool(node_x, batch)), dim=1)

        mu, logvar = self.gaussian_mlp_encoder(graph_x).chunk(2, dim=1)
        #mu = self.gaussian_mlp_encoder(graph_x)
        mu, logvar, z = self.reparameterise(mu, logvar)

        return mu, logvar, z
        # return graph_x

    def get_vae_loss(self, y, x, edge_index, batch, mu, logvar):

        # graph representation
        node_x = self.get_node_rep(x, edge_index)
        graph_x = torch.cat((global_mean_pool(node_x, batch), global_max_pool(node_x, batch)), dim=1)

        # Reconstruction loss
        marginal_likelihood = torch.sum(graph_x * torch.log(y+1e-8) + (1 - graph_x) * torch.log(1 - y + 1e-8), dim=1)
        # marginal_likelihood = torch.mean(marginal_likelihood)

        # KL Divergence
        kl_divergence = 0.5 * torch.sum(mu**2 + logvar**2 - torch.log(1e-8 + logvar**2) - 1, dim=1)
        # kl_divergence = torch.mean(kl_divergence)

        return marginal_likelihood, kl_divergence

def get_disentangle_loss(mu_i, mu_j, y_i, y_j):
    # Disentagle loss
    # First determine whether i, j belong to the same class
    vector_y = torch.mean(torch.pow(y_j.float() - y_i.float(), 2))
    vector_mu = torch.mean(torch.pow(mu_j - mu_i, 2), dim=1)
    loss_bac = 60 * vector_y

    loss_0 = torch.mean(torch.multiply(vector_mu, 1 - vector_y))
    loss_1 = torch.mean(torch.multiply(torch.abs(F.relu(loss_bac-vector_mu)), vector_y))

    disentagle_loss = loss_0 + loss_1

    return disentagle_loss