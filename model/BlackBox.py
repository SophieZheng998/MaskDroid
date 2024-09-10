import sys
import json
import numpy as np
from tqdm import tqdm
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from model.base.abstract_run import AbstractRUN
from model.GNN import GNN_RUN, GNN
from model.GNN_v2 import GNN_v2
from model.PreModel import PreModel
from model.PreModel_v3 import PreModel_v3
from model.PreModel_v4 import PreModel_v4
from model.MsDroid import MsDroid
from model.GNN_VAE import GNN_VAE
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
from torch_geometric.nn.conv import GINConv, SAGEConv, GATConv, GCNConv
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import os



class BlackBox_RUN(GNN_RUN):
    ### PreModel_v3
    #def __init__(self, args, checkpoint_path = "../weights/PreModel_v3/year=all_lr=0.001_h=64_drop=0.2_ms=0.8/epoch=10.checkpoint.pth.tar"):
    #def __init__(self, args, checkpoint_path = "/storage_fast/jnzheng/GCL_security/weights/PreModel_v3/year=2016-2019_lr=0.001_h=64_drop=0.2_ms=0.8/epoch=7.checkpoint.pth.tar"):
    ### FDVAE
    #def __init__(self, args, checkpoint_path = "../weights/GNN_VAE/year=all_lr=0.001_h=64_drop=0.2_ms=0.8/epoch=8.checkpoint.pth.tar"):
    #def __init__(self, args, checkpoint_path = "/storage_fast/jnzheng/GCL_security/weights/GNN_VAE/year=2016-2019_lr=0.001_h=64_drop=0.2_ms=0.8/epoch=0.checkpoint.pth.tar"):
    ### Msdroid
    #def __init__(self, args, checkpoint_path = "../weights/MsDroid/year=all_lr=0.001_h=64_drop=0.2_ms=0.8/epoch=1.checkpoint.pth.tar"):
    def __init__(self, args, checkpoint_path = "/storage_fast/jnzheng/GCL_security/weights/MsDroid/year=2016-2019_lr=0.001_h=64_drop=0.2_ms=0.8/epoch=1.checkpoint.pth.tar"):
        super().__init__(args)

        self.checkpoint_path = checkpoint_path
        self.blackmodel = None

        self.blackboxtype = args.blackboxtype

        if args.blackboxtype == "GNN":
            self.blackmodel = GNN(args)
        elif args.blackboxtype == 'GNN_v2':
            self.blackmodel = GNN_v2(args)
        elif args.blackboxtype == 'GNN_VAE':
            self.blackmodel = GNN_VAE(args)
        elif args.blackboxtype == "PreModel":
            self.blackmodel = PreModel(args)
        elif args.blackboxtype == 'PreModel_v3':
            self.blackmodel = PreModel_v3(args)
        elif args.blackboxtype == 'PreModel_v4':
            self.blackmodel = PreModel_v4(args)
        elif args.blackboxtype == 'MsDroid':
            self.blackmodel = MsDroid(args)

        self.blackmodel.load_state_dict(torch.load(checkpoint_path, map_location="cuda:{}".format(args.cuda))['state_dict'])
        self.blackmodel.eval()
        self.blackmodel.cuda(self.device)

        # TODO
        with open("/storage_fast/jnzheng/GCL_security/data/ASR/label_dict_rebuttal.json", "r") as f:
            self.label_dictionary = json.load(f)

    def train_one_epoch(self, epoch):

        # TODO : debug
        #is_best, temp_flag, _  = self.evaluation(self.args, self.data, self.model, epoch, self.base_path)
        
        #train_data = self.data.construct_dataset(mode = "train")
        train_idx = list(range(self.data.n_train))

        train_loader = DataLoader(train_idx, batch_size=self.batch_size, shuffle=True, worker_init_fn=np.random.seed(101))
        
        self.model.train()

        running_loss, num_batches = 0, 0

        pbar = tqdm(enumerate(train_loader), mininterval=2, total = len(train_loader))

        for batch_idx, data in pbar:

            train_objects, data_paths = self.data.construct_dataset(data, "train")

            data = Batch.from_data_list(train_objects)
            
            data.cuda(self.device)

            #if 1:
            self.optimizer.zero_grad()

            if self.blackboxtype == 'GNN':
                teacher_labels = self.blackmodel.evaluate(data.x, data.edge_index, data.batch)
            elif self.blackboxtype == 'GNN_v2':
                teacher_labels, _ = self.blackmodel.evaluate(data.x, data.edge_index, data.batch)
            elif self.blackboxtype == 'PreModel':
                teacher_labels, _, _ = self.blackmodel.evaluate(data.x, data.edge_index, data.batch)
            elif self.blackboxtype == 'PreModel_v3' or self.blackboxtype == 'PreModel_v4':
                teacher_labels, _, _ = self.blackmodel.evaluate(data.x, data.edge_index, data.batch)
            elif self.blackboxtype == 'GNN_VAE' :
                teacher_labels, _ = self.blackmodel.evaluate(data.x, data.edge_index, data.batch)
            elif self.blackboxtype == 'MsDroid':
                teacher_labels = []
                for data_name in data_paths:
                    teacher_labels.append(self.label_dictionary[data_name])

            _, pred_loss = self.model(data.x, data.edge_index, data.batch, torch.tensor(teacher_labels).to(self.device))
                
            loss = pred_loss
        
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            num_batches += 1
 
        
        return [running_loss/num_batches]
 
    def evaluation(self, args, data, model, epoch, base_path, name = "test"):

        if name == "valid":
            evaluate_idx = list(range(self.data.n_valid))
        else:
            evaluate_idx = list(range(self.data.n_test))
            
        eval_loader = DataLoader(evaluate_idx, batch_size=self.batch_size, shuffle=True, worker_init_fn=np.random.seed(101))

        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            self.model.train()
            for batch in eval_loader:

                evaluate_objects, data_paths = self.data.construct_dataset(batch, "test")
                
                batch = Batch.from_data_list(evaluate_objects)
                batch.cuda(self.device)
                logits, pred_loss = model(batch.x, batch.edge_index, batch.batch, batch.y)
                predictions = logits.argmax(dim=1)

                if self.blackboxtype == 'GNN':
                    teacher_labels = self.blackmodel.evaluate(batch.x, batch.edge_index, batch.batch)
                elif self.blackboxtype == 'GNN_VAE':
                    teacher_labels, _ = self.blackmodel.evaluate(batch.x, batch.edge_index, batch.batch)
                elif self.blackboxtype == 'GNN_v2':
                    teacher_labels, _ = self.blackmodel.evaluate(batch.x, batch.edge_index, batch.batch)
                elif self.blackboxtype == 'PreModel':
                    teacher_labels, _, _ = self.blackmodel.evaluate(batch.x, batch.edge_index, batch.batch)
                elif self.blackboxtype == 'PreModel_v3' or self.blackboxtype == 'PreModel_v4':
                    teacher_labels, _ , _ = self.blackmodel.evaluate(batch.x, batch.edge_index, batch.batch)
                elif self.blackboxtype == 'MsDroid':
                    teacher_labels = []
                    for data_name in data_paths:
                        teacher_labels.append(self.label_dictionary[data_name])
                    # result_logits, pred_loss = self.blackmodel(batch_black.x, batch_black.edge_index, batch_black.batch, batch_black.y)
                    # teacher_labels = result_logits.values.argmax().item()
            
                if self.blackboxtype == 'MsDroid':
                    true_labels.extend(teacher_labels)
                else:
                    true_labels.extend(teacher_labels.cpu().tolist())

                predicted_labels.extend(predictions.cpu().tolist())

        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        accuracy = accuracy_score(true_labels, predicted_labels)

        n_ret = {"precision": precision, "recall": recall, "f1_score": f1, "accuracy": accuracy}

        perf_str = name+':{}'.format(n_ret)
        print(perf_str)
        with open(base_path + 'stats.txt', 'a') as f:
            f.write(perf_str + "\n")

        # Check if need to early stop (on validation)
        is_best=False
        early_stop=False

        if name=="test":
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

        # the whole pipeline of the training process

class BlackBox(GNN):

    def __init__(self, args):
        super().__init__(args)
        self.blackboxtype = args.blackboxtype

    def forward(self, x, edge_index, batch, y):
        # node representation
        node_x = self.get_node_rep(x, edge_index)

        # graph representation
        graph_x = torch.cat((global_mean_pool(node_x, batch), global_max_pool(node_x, batch)), dim=1)

        # graph classification
        logits = self.graph_pred_linear(graph_x)

        #y = y.to(torch.float32)
        y = y.to(torch.int64)

        # if self.blackboxtype == 'MsDroid':
        #     pred_loss = self.criterion(logits, y.view(1))
        # else:
        pred_loss = self.criterion(logits, y)

        return logits, pred_loss
    