#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
This file defines the Mu-Sigma-Multi-Layer Perceptron (MU-SIGMA-MLP).
"""
import torch
import torch.nn as nn
import time
import numpy as np
import sys
from logging import Logger
from model.Utils.Network.helper import get_device, adjust_learning_rate, eval_metrics
from model.FD_VGAE import FD_VGAE
from tqdm import tqdm
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
from model.Utils.Network.helper import get_device, adjust_learning_rate, eval_metrics



class VGAE_MLP_RUN(AbstractRUN):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.seed = args.seed
        self.pretrain_vae = FD_VGAE(args)
        self.max_loss = np.inf

        self.checkpoint_path = "../weights/FD_VGAE/type=gcn_lr=0.001_h=64_drop=0.2_JK=last_ms=0.8/epoch=7.checkpoint.pth.tar"
        self.pretrain_vae.load_state_dict(torch.load(self.checkpoint_path, map_location="cuda:{}".format(args.cuda))['state_dict'])
        self.pretrain_vae.eval()
        self.pretrain_vae.cuda(self.device)

    def train_one_epoch(self, epoch):
        
        #train_data = self.data.construct_dataset(mode = "train")
        train_idx = list(range(self.data.n_train))

        train_loader = DataLoader(train_idx, batch_size=self.batch_size, shuffle=True, worker_init_fn=np.random.seed(self.seed))
        
        self.model.train()

        running_loss, num_batches = 0, 0

        pbar = tqdm(enumerate(train_loader), mininterval=2, total = len(train_loader))

        adjust_learning_rate(self.optimizer, epoch, self.lr)

        
        for batch_idx, data in pbar:

            # access data 
            train_objects = self.data.construct_dataset(data, "train")

            data = Batch.from_data_list(train_objects)

            data.cuda(self.device)

            # Forward
            z, mu, logvar = self.pretrain_vae(data.x, data.edge_index)

            mu_sigma = torch.cat((mu, logvar), dim=1)

            loss = self.model(mu_sigma, data.batch, data.y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()  
           
            running_loss += loss.detach().item()
            
            num_batches += 1

        return [running_loss/num_batches]
    
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

        # load in vae model
        
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

        true_labels = []
        y_pred = []
    
        with torch.no_grad():
            self.model.train()
            for batch in eval_loader:
                #batch = [x.cuda(self.device) for x in batch]
                evaluate_objects = self.data.construct_dataset(batch, name)
                batch = Batch.from_data_list(evaluate_objects)

                batch.cuda(self.device)

                mu, sigma = self.pretrain_vae.get_encoder_output(batch.x, batch.edge_index)

                mu_sigma = torch.cat((mu, sigma), dim=1)
                y_pred_x = model.predict(mu_sigma, batch.batch)
                y_pred.append(y_pred_x)
                true_labels.extend(batch.y.cpu().tolist())

        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
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


class VGAE_MLP(nn.Module):
    def __init__(self, args, in_channels: int=64, hidden_channels: int=16, out_channels: int=2, dropout_ratio: float = 0.1):
        super(VGAE_MLP, self).__init__()

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

        self.pretrain_vae = FD_VGAE(args)
        self.checkpoint_path = "../weights/FD_VGAE/type=gcn_lr=0.001_h=64_drop=0.2_JK=last_ms=0.8/epoch=7.checkpoint.pth.tar"

        self.pretrain_vae.load_state_dict(torch.load(self.checkpoint_path, map_location="cuda:{}".format(args.cuda))['state_dict'])
        self.pretrain_vae.eval()
        self.pretrain_vae.cuda(args.cuda)

        #for param in self.pretrain_vae.parameters():
        #   param.requires_grad = False

    def forward(self, mu_sigma, batch, y):

        graph_x = global_mean_pool(mu_sigma, batch)

        y_score = self.mu_sigma_linear(graph_x)
        y = torch.nn.functional.one_hot(y, num_classes=2).float()
        loss = self.loss_fn(y_score, y)

        return loss
    
    def predict(self, mu_sigma, batch):

        graph_x = global_mean_pool(mu_sigma, batch)

        y_score = self.mu_sigma_linear(graph_x)
        y_pred = torch.argmax(y_score, dim=1)

        return y_pred

    def evaluate(self, x, edge_index, batch):

        mu, sigma, z = self.pretrain_vae(x, edge_index)
        mu_sigma = torch.cat((mu, sigma), dim=1)
        graph_x = global_mean_pool(mu_sigma, batch)
        y_score = self.mu_sigma_linear(graph_x)
        y_pred = torch.argmax(y_score, dim=1)
        
        return y_pred, y_score

    