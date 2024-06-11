import sys
import numpy as np
from tqdm import tqdm
import torch
import time
import json
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from model.base.abstract_run import AbstractRUN
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
from torch_geometric.nn.conv import GINConv, SAGEConv, GATConv, GCNConv
from model.FD_VAE import FD_VAE
#from torch.cuda.amp import autocast, GradScaler


class GNN_VAE_RUN(AbstractRUN):

    def __init__(self, args) -> None:
        super().__init__(args)
        self.contrastive = args.contrastive
        self.seed = args.seed

        
    def train_one_epoch(self, epoch):
        
        #train_data = self.data.construct_dataset(mode = "train")
        train_idx = list(range(self.data.n_train))

        train_loader = DataLoader(train_idx, batch_size=self.batch_size, shuffle=True, worker_init_fn=np.random.seed(self.seed))
        
        self.model.train()

        running_loss, running_pred_loss, running_cl_loss, num_batches = 0, 0, 0, 0

        pbar = tqdm(enumerate(train_loader), mininterval=2, total = len(train_loader))

        
        for _, data in pbar:

            # access data 
            train_objects, _ = self.data.construct_dataset(data, "train")
            data = Batch.from_data_list(train_objects)
            data.cuda(self.device)
            
            self.optimizer.zero_grad()
            
            if self.contrastive:
                _, pred_loss, cl_loss = self.model(data.x, data.edge_index, data.batch, data.y)
                loss = pred_loss + cl_loss
            else:
               
                _, pred_loss = self.model(data.x, data.edge_index, data.batch, data.y)
                loss = pred_loss
           
            loss.backward()
            self.optimizer.step()  
           
            running_loss += loss.detach().item()
            running_pred_loss += pred_loss.detach().item()
            
            if self.contrastive:
                running_cl_loss += cl_loss.detach().item()

            num_batches += 1

        if self.contrastive:
            return [running_loss/num_batches, running_pred_loss/num_batches, running_cl_loss/num_batches]
        else:
            return [running_loss/num_batches]


class GNN_VAE(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        
        # GNN parameters
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device(args.cuda)
        self.num_layers = args.num_layers
        self.in_channels = args.in_channels
        self.in_channels_cp = args.in_channels_cp
        self.hidden_channels = args.hidden_channels
        self.mid_channels = args.mid_channels
        self.out_channels = args.out_channels
        self.dropout_ratio = args.dropout_ratio
        self.gnn_type = args.gnn_type
        self.JK = args.JK
        self.eps = args.eps
        self.batch_size = args.batch_size

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.pretrain_vae = FD_VAE(args)
        self.pretrain_vae = self.pretrain_vae.to(self.device)
        self.pretrain_vae.train()

        # message passing layers
        if self.num_layers < 1:
            raise ValueError('Number of GNN layers must be greater than 0.')
        for layer in range(self.num_layers):
            if layer == 0:
                self.convs.append(self.gnn_layer(self.gnn_type, self.in_channels, self.hidden_channels))
            else:
                self.convs.append(self.gnn_layer(self.gnn_type, self.hidden_channels, self.hidden_channels))
            self.batch_norms.append(BatchNorm(self.hidden_channels))

        # graph classification layer (input graph representation: global_mean_pool + global_max_pool)
        
        graph_channels = self.hidden_channels * (self.num_layers + 1) if self.JK == 'concat' else self.hidden_channels
        fc1=nn.Linear(graph_channels * 2, self.hidden_channels)
        fc2=nn.Linear(self.hidden_channels, self.out_channels)
        nn.init.xavier_uniform_(fc1.weight)
        nn.init.xavier_uniform_(fc2.weight)
        self.graph_pred_linear = nn.Sequential(
            #nn.Linear(graph_channels * 2, self.hidden_channels),
            fc1,
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            fc2,
            #nn.Linear(self.hidden_channels, self.out_channels)
        )

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
        node_x_cl = 0
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

    def forward(self, x, edge_index, batch, y):

        # node_x = self.get_node_rep(x, edge_index)

        # # graph representation
        # graph_x = torch.cat((global_mean_pool(node_x, batch), global_max_pool(node_x, batch)), dim=1)

        mu, sigma, z = self.pretrain_vae.get_encoder_output(x, edge_index, batch)
        mu_sigma = torch.cat((mu, sigma), dim=1)

        # graph classification
        logits = self.graph_pred_linear(mu_sigma)
        
        pred_loss = self.criterion(logits, y)

        return logits, pred_loss
    
    def evaluate(self, x, edge_index, batch):
        # node representation
        # node_x = self.get_node_rep(x, edge_index)

        # # graph representation
        # graph_x = torch.cat((global_mean_pool(node_x, batch), global_max_pool(node_x, batch)), dim=1)
        mu, sigma, z = self.pretrain_vae.get_encoder_output(x, edge_index, batch)
        mu_sigma = torch.cat((mu, sigma), dim=1)

        # graph classification
        logits = self.graph_pred_linear(mu_sigma)
        predicted_labels = logits.argmax(dim=1)
        
        return predicted_labels, logits
    