# mlp
import sys
import numpy as np
from tqdm import tqdm
import time
from functools import partial
import torch
import pandas as pd
import torch.nn as nn
from model.base.utils import *
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from model.base.abstract_run import AbstractRUN
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


class PreModel_v4_RUN(AbstractRUN):

    def __init__(self, args) -> None:
        super().__init__(args)
        self.loss_pretrain = np.inf
        self.need_pretrain = args.need_pretrain
        self.seed = args.seed

    def execute(self):

        # write args
        perf_str = str(self.args)
        with open(self.base_path + 'stats.txt','a') as f:
            f.write(perf_str+"\n")

        self.model, self.start_epoch = self.restore_checkpoint(self.model, self.base_path, self.device) 

        if self.need_pretrain:
            print("start training") 
            self.train()
            self.model = self.restore_best_checkpoint(self.data.best_valid_epoch, self.model, self.base_path, self.device)
        
        print("start evaluation")
        self.model.eval() # evaluate the best model
        print_str = "The best epoch is % d" % self.data.best_valid_epoch
        with open(self.base_path +'stats.txt', 'a') as f:
            f.write(print_str + "\n")

        # use threshold to get the result of validation data / test data
        # use validation result to choose the best threshold, and record the test resul
        #print("check training data representation")
        #_, _, final_train = self.evaluation(self.args, self.data, self.model, self.data.best_valid_epoch, self.base_path, name = "train")
        #_, _, final_valid = self.evaluation(self.args, self.data, self.model, self.data.best_valid_epoch, self.base_path, name = "valid")
        _, _, final_test = self.evaluation(self.args, self.data, self.model, self.data.best_valid_epoch, self.base_path, name = "test")
        
    # define the training process
    def train(self) -> None:
        
        self.set_optimizer() # get the optimizer
        self.flag = False

        for epoch in range(self.start_epoch, self.max_epoch):
            # print(self.model.embed_user.weight)
            if self.flag: # early stop
                break
            # All models
            t1=time.time()
            losses, loss_batch = self.train_one_epoch(epoch) # train one epoch
            t2=time.time()
            self.document_running_loss(losses, epoch, t2-t1) # report the loss
            if (epoch + 1) % self.verbose == 0: # evaluate the model
                self.eval_and_check_early_stop(epoch, loss_batch)     

    def train_one_epoch(self, epoch):

        #train_data = self.data.construct_dataset(mode = "train")
        train_idx = list(range(self.data.n_train))

        train_loader = DataLoader(train_idx, batch_size=self.batch_size, shuffle=True, worker_init_fn=np.random.seed(self.seed))
        
        self.model.train()

        running_loss, running_rec_loss, running_mlp_loss, num_batches = 0, 0, 0, 0,

        pbar = tqdm(enumerate(train_loader), mininterval=2, total = len(train_loader))
        
        for _, data in pbar:

            # access data 

            train_objects, _ = self.data.construct_dataset(data, "train")

            data = Batch.from_data_list(train_objects)

            data.cuda(self.device)

            self.optimizer.zero_grad()
            rec_loss, mlp_loss,  = self.model(data.x, data.edge_index, data.batch, data.y)
            loss = rec_loss + mlp_loss
            
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            running_rec_loss += rec_loss.detach().item()
            running_mlp_loss += mlp_loss.detach().item()

            num_batches += 1
        
        return [running_loss/num_batches, running_rec_loss/num_batches, running_mlp_loss/num_batches], running_loss      

    def eval_and_check_early_stop(self, epoch, loss_batch):
        self.model.eval()

        is_best, temp_flag, _  = self.evaluation(self.args, self.data, self.model, epoch, self.base_path, name = "valid")
            
        if is_best:
            save_checkpoint(self.model, epoch, self.base_path, self.checkpoint_buffer, self.args.max2keep)

        if temp_flag:
            self.flag = True

        self.model.train()
  
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
                evaluate_objects, _ = self.data.construct_dataset(batch, name)
                batch = Batch.from_data_list(evaluate_objects)

                batch.cuda(self.device)
                predictions, logits_mlp = model.evaluate(batch.x, batch.edge_index, batch.batch)
                
                true_labels.extend(batch.y.cpu().tolist())
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


class PreModel_v4(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        
        # GNN parameters
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_rec = self.setup_loss_fn(loss_fn = 'sce', alpha_l = 2)
        self.device = torch.device(args.cuda)
        self.num_layers = args.num_layers
        self.in_channels = args.in_channels
        self.in_channels_cp = args.in_channels_cp
        self.hidden_channels = args.hidden_channels
        self.out_channels = args.out_channels
        self.dropout_ratio = args.dropout_ratio
        self.batch_size = args.batch_size
        self.JK = args.JK
        self.lambda_rec = args.lambda_rec

        # benign prototype
        #self.v_prototype = nn.parameter.Parameter(torch.zeros(1, self.hidden_channels), requires_grad = True)
        #self.v_prototype_mal = nn.parameter.Parameter(torch.zeros(1, self.hidden_channels), requires_grad = True)
        #nn.init.xavier_uniform_(self.v_prototype)
        #nn.init.xavier_uniform_(self.v_prototype_mal)
        self.lambda_cl = args.lambda_cl
        self.thres = args.thres

        # GraphMAE
        self.preprocess = nn.Linear(args.in_channels, self.in_channels_cp, bias=False)
        self._mask_rate = args.mask_rate
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.in_channels_cp))
        self.encoder = EDcoder(args, self.in_channels_cp, self.hidden_channels)
        self.encoder_to_decoder = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.decoder = EDcoder(args, self.hidden_channels, self.in_channels_cp)

        graph_channels = self.hidden_channels * (self.num_layers + 1) if self.JK == 'concat' else self.hidden_channels
        #self.graph_pred_linear_v2 = nn.Linear(graph_channels * 2, self.hidden_channels)
        self.graph_pred_linear = nn.Sequential(
            nn.Linear(graph_channels * 2, self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.hidden_channels, self.out_channels)
        )
       
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

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

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
    
        # mlp loss
        logits = self.graph_pred_linear(graph_x)
        mlp_loss = self.criterion(logits, y)

        return mlp_loss, rec_loss
    
    def evaluate(self, x, edge_index, batch):

        out_x = self.preprocess(x)
        
        enc_rep = self.encoder(out_x, edge_index)

        graph_x = torch.cat((global_mean_pool(enc_rep, batch), global_max_pool(enc_rep, batch)), dim=1)

        # mlp classifier
        logits_mlp = self.graph_pred_linear(graph_x)
        predicted_labels = logits_mlp.argmax(dim=1)

        return predicted_labels, logits_mlp, graph_x


class EDcoder(torch.nn.Module):
    
    def __init__(self, args, in_channels, out_channels):
        super().__init__()
        
        # GNN parameters
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device(args.cuda)
        self.num_layers = args.num_layers
        self.in_channels = in_channels
        self.hidden_channels = args.hidden_channels
        self.out_channels = out_channels
        self.dropout_ratio = args.dropout_ratio
        self.gnn_type = args.gnn_type
        self.JK = args.JK
        self.batch_size = args.batch_size

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # message passing layers
        if self.num_layers < 1:
            raise ValueError('Number of GNN layers must be greater than 0.')
        for layer in range(self.num_layers):
            if layer == 0:
                self.convs.append(self.gnn_layer(self.gnn_type, self.in_channels, self.hidden_channels))
            else:
                self.convs.append(self.gnn_layer(self.gnn_type, self.hidden_channels, self.hidden_channels))
            self.batch_norms.append(BatchNorm(self.hidden_channels))

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
            #x = F.dropout(x, self.dropout_ratio, training=False)
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

    def forward(self, x, edge_index):
        # node representation
        node_x = self.get_node_rep(x, edge_index)

        return node_x
    


 