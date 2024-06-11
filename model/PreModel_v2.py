# mlp + infonce + margin
import sys
import numpy as np
from tqdm import tqdm
import time
from functools import partial
import torch
import math
import pandas as pd
import torch.nn as nn
from model.base.utils import *
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from model.PreModel import PreModel_RUN, PreModel, EDcoder, sce_loss
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
from torch_geometric.nn.conv import GINConv, SAGEConv, GATConv, GCNConv
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

class PreModel_v2_RUN(PreModel_RUN):

    def __init__(self, args) -> None:
        super().__init__(args)
        self.seed = args.seed
        
    def train_one_epoch(self, epoch):
        
        #train_data = self.data.construct_dataset(mode = "train")
        train_idx = list(range(self.data.n_train))

        train_loader = DataLoader(train_idx, batch_size=self.batch_size, shuffle=True, worker_init_fn=np.random.seed(self.seed))
        
        self.model.train()

        #running_loss, running_cl_loss, running_rec_loss, running_mlp_loss, num_batches = 0, 0, 0, 0, 0
        running_loss, running_rec_loss, running_mlp_loss, num_batches = 0, 0, 0, 0, 

        pbar = tqdm(enumerate(train_loader), mininterval=2, total = len(train_loader))
        
        for _, data in pbar:

            # access data 

            train_objects = self.data.construct_dataset(data, "train")

            data = Batch.from_data_list(train_objects)

            data.cuda(self.device)

            self.optimizer.zero_grad()
            rec_loss, mlp_loss = self.model(data.x, data.edge_index, data.batch, data.y)
            #cl_loss, rec_loss, mlp_loss = self.model(data.x, data.edge_index, data.batch, data.y)
            #loss = cl_loss + rec_loss + mlp_loss
            loss = rec_loss + mlp_loss
            
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            #running_cl_loss += cl_loss.detach().item()
            running_rec_loss += rec_loss.detach().item()
            running_mlp_loss += mlp_loss.detach().item()

            num_batches += 1

        return [running_loss/num_batches, running_rec_loss/num_batches, running_mlp_loss/num_batches], running_loss
            #return [running_loss/num_batches, running_cl_loss/num_batches, running_rec_loss/num_batches, running_mlp_loss/num_batches], running_loss
    
    def evaluation(self, args, data, model, epoch, base_path, name = "valid"):

        if name == "train":
            evaluate_idx = list(range(self.data.n_train))
        elif name == "valid":
            evaluate_idx = list(range(self.data.n_valid))
        else:
            evaluate_idx = list(range(self.data.n_test))
            
        eval_loader = DataLoader(evaluate_idx, batch_size=self.batch_size, shuffle=False)

        true_labels = []
        predicted_labels = []
    
        with torch.no_grad():
            for batch in eval_loader:
                #batch = [x.cuda(self.device) for x in batch]
                evaluate_objects = self.data.construct_dataset(batch, name)
                batch = Batch.from_data_list(evaluate_objects)

                batch.cuda(self.device)
                predictions, _ = model.evaluate(batch.x, batch.edge_index, batch.batch)
               
                true_labels.extend(batch.y.cpu().tolist())
                predicted_labels.extend(predictions.cpu().tolist())
               
        
        f1 = f1_score(true_labels, predicted_labels)
        accuracy = accuracy_score(true_labels, predicted_labels)
        confusion = confusion_matrix(true_labels, predicted_labels)

        false_alarm_rate = confusion[0, 1] / (confusion[0, 0] + confusion[0, 1])
        miss_detection_rate = confusion[1, 0] / (confusion[1, 0] + confusion[1, 1])

        n_ret = {"false_alarm": false_alarm_rate, "miss_detection": miss_detection_rate, "f1_score": f1, "accuracy": accuracy}

        perf_str = name+':{}'.format(n_ret)
        print(perf_str)
        with open(base_path + 'stats.txt', 'a') as f:
            f.write(perf_str + "\n")


        # Check if need to early stop (on validation)
        is_best=False
        early_stop=False

        if name=="valid":
            if f1 > data.best_valid_f1:
                data.best_valid_epoch = epoch
                data.best_valid_f1 = f1
                data.patience = 0
                is_best=True
            else:
                data.patience += 1
                if data.patience >= args.patience:
                    print_str = "The best performance epoch is % d " % data.best_valid_epoch
                    print(print_str)
                    early_stop=True

        return is_best, early_stop, n_ret


class PreModel_v2(PreModel):

    def __init__(self, args):
        super().__init__(args)
        self.tau = args.tau
        self.margin = args.margin


    def encoding_mask_noise(self, x, mask_rate=0.3):
        
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=self.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        #keep_nodes = perm[num_mask_nodes: ]

        out_x_init = self.preprocess(x)
        out_x = out_x_init.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return out_x_init, out_x, mask_nodes, #keep_nodes)

    def forward(self, x, edge_index, batch, y):

        # calculate reconstruction loss
        out_x, out_x_init, mask_nodes = self.encoding_mask_noise(x, self._mask_rate)

        enc_rep = self.encoder(out_x, edge_index)

        enc_rep_init = self.encoder(out_x_init, edge_index)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        rep_init = self.encoder_to_decoder(enc_rep_init)

        rep[mask_nodes] = 0

        recon = self.decoder(rep, edge_index)

        #recon_init = self.decoder(rep_init, edge_index)

        x_init = out_x_init[mask_nodes]
        x_rec = recon[mask_nodes]

        rec_loss = self.lambda_rec * self.criterion_rec(x_rec, x_init)

        # calculate classification loss
        graph_x = torch.cat((global_mean_pool(enc_rep, batch), global_max_pool(enc_rep, batch)), dim=1)
        
        graph_x_init = torch.cat((global_mean_pool(enc_rep_init, batch), global_max_pool(enc_rep_init, batch)), dim=1)

        graph_rep = self.graph_pred_linear_v2(graph_x)

        graph_rep_init = self.graph_pred_linear_v2(graph_x_init)

        # compute 
        features = nn.functional.normalize(graph_rep, dim=1, p=2)
        augmented_features = nn.functional.normalize(graph_rep_init, dim=1, p=2)

        similarity_matrix = torch.matmul(features, augmented_features.T)
        #class_mask = ~(y.unsqueeze(0) != y.unsqueeze(1))

        positive_similarity = torch.diag(similarity_matrix)

        angles = torch.arccos(torch.clamp(positive_similarity, -1 + 1e-7, 1 - 1e-7)) + self.margin

        positive_similarity = torch.cos(torch.min(angles, math.pi*torch.ones_like(angles)))

        numerator = torch.exp(positive_similarity / self.tau)
        denominator = torch.sum(torch.exp(similarity_matrix / self.tau), dim = 1)

        cl_loss = self.lambda_cl * torch.mean(torch.negative(torch.log(numerator/denominator)))

        # mlp loss
        logits = self.graph_pred_linear(graph_x)
        mlp_loss = self.criterion(logits, y)

        #return cl_loss, rec_loss, mlp_loss, 
        return rec_loss, mlp_loss, 
    
    def evaluate(self, x, edge_index, batch):

        out_x = self.preprocess(x)
        
        enc_rep = self.encoder(out_x, edge_index)

        graph_x = torch.cat((global_mean_pool(enc_rep, batch), global_max_pool(enc_rep, batch)), dim=1)
        
        # mlp classifier
        logits_mlp = self.graph_pred_linear(graph_x)
        predicted_labels = logits_mlp.argmax(dim=1)

        return predicted_labels,  logits_mlp


 