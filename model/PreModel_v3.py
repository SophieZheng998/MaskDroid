# proxy 
import sys
import json
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
from model.PreModel import PreModel_RUN, PreModel, EDcoder
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
from torch_geometric.nn.conv import GINConv, SAGEConv, GATConv, GCNConv
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

def sce_loss(x, y, alpha=2):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


class PreModel_v3_RUN(PreModel_RUN):

    def __init__(self, args) -> None:
        super().__init__(args)
        self.seed = args.seed
        
    def train_one_epoch(self, epoch):
        
        #train_data = self.data.construct_dataset(mode = "train")
        train_idx = list(range(self.data.n_train))

        train_loader = DataLoader(train_idx, batch_size=self.batch_size, shuffle=True, worker_init_fn=np.random.seed(self.seed))
        
        self.model.train()

        #running_loss, running_cl_loss, running_rec_loss, running_mlp_loss, num_batches = 0, 0, 0, 0, 0
        running_loss, running_rec_loss, running_cl_loss, num_batches = 0, 0, 0, 0, 

        pbar = tqdm(enumerate(train_loader), mininterval=2, total = len(train_loader))
        
        for _, data in pbar:

            # access data 
            
            train_objects, _ = self.data.construct_dataset(data, "train")

            data = Batch.from_data_list(train_objects)

            data.cuda(self.device)

            self.optimizer.zero_grad()
            cl_loss, rec_loss, = self.model(data.x, data.edge_index, data.batch, data.y)
            loss = rec_loss + cl_loss
            
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            #running_cl_loss += cl_loss.detach().item()
            running_cl_loss += cl_loss.detach().item()
            running_rec_loss += rec_loss.detach().item()


            num_batches += 1

        return [running_loss/num_batches, running_cl_loss/num_batches, running_rec_loss/num_batches], running_loss
            
    def evaluation(self, args, data, model, epoch, base_path, name = "valid"):

        if name == "train":
            evaluate_idx = list(range(self.data.n_train))
        elif name == "valid":
            evaluate_idx = list(range(self.data.n_valid))
        else:
            evaluate_idx = list(range(self.data.n_test))
            
        eval_loader = DataLoader(evaluate_idx, batch_size=self.batch_size, shuffle=True, worker_init_fn=np.random.seed(self.seed))

        true_labels = []
        predicted_labels = []
        names = []
    
        with torch.no_grad():
            for batch in eval_loader:
                #batch = [x.cuda(self.device) for x in batch]
                evaluate_objects, datapath_list = self.data.construct_dataset(batch, name)
                batch = Batch.from_data_list(evaluate_objects)

                batch.cuda(self.device)
                predictions, _, _ = model.evaluate(batch.x, batch.edge_index, batch.batch)
               
                true_labels.extend(batch.y.cpu().tolist())
                predicted_labels.extend(predictions.cpu().tolist())

                if name == "test" and self.need_record:
                    for true_label, predicted_label, datapath in zip(batch.y.cpu().tolist(), predictions.cpu().tolist(), datapath_list):
                        
                        if true_label == 1 and predicted_label == 1:
                            names.append(datapath)

        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        accuracy = accuracy_score(true_labels, predicted_labels)

        n_ret = {"precision": precision, "recall": recall, "f1_score": f1, "accuracy": accuracy}

        perf_str = name+':{}'.format(n_ret)
        print(perf_str)
        with open(base_path + 'stats.txt', 'a') as f:
            f.write(perf_str + "\n")

        if name == "test" and self.need_record:
            with open(base_path + '_names.json', 'w') as f:
                json.dump(names, f)

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


class PreModel_v3(PreModel):

    def __init__(self, args):
        super().__init__(args)
        self.tau = args.tau
        self.margin = args.margin
        self.test_mask_rate = args.test_mask_rate

    def encoding_mask_noise(self, x, mask_rate=0.3):
        
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=self.device)
        
        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        out_x_init = self.preprocess(x)
        out_x = out_x_init.clone()
        out_x[mask_nodes] = self.enc_mask_token

        return out_x_init, out_x, mask_nodes, #keep_nodes)

    def forward(self, x, edge_index, batch, y):
 
        # calculate reconstruction loss
        out_x, out_x_init, mask_nodes = self.encoding_mask_noise(x, self._mask_rate)

        enc_rep = self.encoder(out_x, edge_index)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        rep[mask_nodes] = 0

        recon = self.decoder(rep, edge_index)

        x_init = out_x_init[mask_nodes]
        x_rec = recon[mask_nodes]

        rec_loss = self.lambda_rec * self.criterion_rec(x_rec, x_init)

        # calculate classification loss
        graph_x = torch.cat((global_mean_pool(enc_rep, batch), global_max_pool(enc_rep, batch)), dim=1)
        
        graph_rep = self.graph_pred_linear_v2(graph_x)

        # compute 
        features = nn.functional.normalize(graph_rep, dim=1, p=2)
        prototype_benign = self.v_prototype/self.v_prototype.norm(dim=1, p=2)
        prototype_mal = self.v_prototype_mal/self.v_prototype_mal.norm(dim=1, p=2)

        cosine_sim_benign = F.cosine_similarity(features, prototype_benign, dim=1)
        cosine_sim_mal = F.cosine_similarity(features, prototype_mal, dim=1)

        # calculate the loss with reweighting
        #weight = torch.where(y == 1, torch.tensor(9.0, device=y.device), torch.tensor(1.0, device=y.device))
        #cosine_sim_benign = weight * (torch.square(cosine_sim_benign[y==1]) + torch.square(1 - cosine_sim_benign[y==0]))
        #cosine_sim_mal = weight * (torch.square(cosine_sim_mal[y==0]) + torch.square(1 - cosine_sim_mal[y==1]))

        # benign
        cosine_sim_benign[y==1] = torch.square(cosine_sim_benign[y==1])
        cosine_sim_benign[y==0] = torch.square(1-cosine_sim_benign[y==0])
        cosine_sim_mal[y==0] = torch.square(cosine_sim_mal[y==0])
        cosine_sim_mal[y==1] = torch.square(1-cosine_sim_mal[y==1])

        reweight_term = torch.where(y == 1, torch.tensor(9.0, device=y.device), torch.tensor(1.0, device=y.device))
        cosine_sim_benign_weighted = cosine_sim_benign * reweight_term
        cosine_sim_mal_weighted = cosine_sim_mal * reweight_term

        cl_loss = self.lambda_cl * torch.sum(cosine_sim_benign_weighted) + self.lambda_cl * torch.sum(cosine_sim_mal_weighted)

        return cl_loss, rec_loss,
    
    def evaluate(self, x, edge_index, batch):

        out_x = self.preprocess(x)

        out_x, out_x_init, mask_nodes = self.encoding_mask_noise(x, self.test_mask_rate)

        enc_rep = self.encoder(out_x, edge_index)

        graph_x = torch.cat((global_mean_pool(enc_rep, batch), global_max_pool(enc_rep, batch)), dim=1)
        
        graph_rep = self.graph_pred_linear_v2(graph_x)

        # calculate the distance between the sample and the prototype
        features = nn.functional.normalize(graph_rep, dim=1, p=2)
        prototype_benign = F.normalize(self.v_prototype, dim=1, p=2)
        prototype_mal = F.normalize(self.v_prototype_mal, dim=1, p=2)
        cosine_sim_benign = F.cosine_similarity(features, prototype_benign, dim=1)
        cosine_sim_mal = F.cosine_similarity(features, prototype_mal, dim=1)
    
        logits_cl = torch.stack([cosine_sim_benign, cosine_sim_mal], dim=1)
       
        mask = cosine_sim_benign > cosine_sim_mal
        predicted_labels = torch.logical_not(mask).int()

        # mlp classifier
        #logits_mlp = self.graph_pred_linear(graph_x)
        #predicted_mlp = logits_mlp.argmax(dim=1)

        # centroid distance  ||   mlp classifier
        #predicted_labels = torch.logical_or(predicted_cl, predicted_mlp).int()

        return predicted_labels, logits_cl, features, #logits_mlp


 