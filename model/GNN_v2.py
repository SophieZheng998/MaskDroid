# GNN + proxy
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
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix


class GNN_v2_RUN(AbstractRUN):

    def __init__(self, args) -> None:
        super().__init__(args)
        self.contrastive = args.contrastive
        self.seed = args.seed

    def train_one_epoch(self, epoch):
        
        #train_data = self.data.construct_dataset(mode = "train")
        train_idx = list(range(self.data.n_train))

        train_loader = DataLoader(train_idx, batch_size=self.batch_size, shuffle=True, worker_init_fn=np.random.seed(self.seed))
        
        self.model.train()

        running_loss, num_batches = 0, 0

        pbar = tqdm(enumerate(train_loader), mininterval=2, total = len(train_loader))

        
        for _, data in pbar:

            # access data 
            train_objects, _ = self.data.construct_dataset(data, "train")
            data = Batch.from_data_list(train_objects)
            data.cuda(self.device)
            
            self.optimizer.zero_grad()
            
            loss = self.model(data.x, data.edge_index, data.batch, data.y)
           
            loss.backward()
            self.optimizer.step()  
           
            running_loss += loss.detach().item()
          

            num_batches += 1

        return [running_loss/num_batches]

    def evaluation(self, args, data, model, epoch, base_path, name = "valid"):

        if name == "valid":
            evaluate_idx = list(range(self.data.n_valid))
        elif name == "test":
            evaluate_idx = list(range(self.data.n_test))
        elif name == "test_v2":
            evaluate_idx = list(range(self.data.n_test_v2))
            
        eval_loader = DataLoader(evaluate_idx, batch_size=self.batch_size, shuffle=False)

        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for batch in eval_loader:
               
                evaluate_objects, _ = self.data.construct_dataset(batch, name)
                batch = Batch.from_data_list(evaluate_objects)

                batch.cuda(self.device)
                predictions, _  = model.evaluate(batch.x, batch.edge_index, batch.batch)
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



class GNN_v2(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        
        # GNN parameters
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device(args.cuda)
        self.num_layers = args.num_layers
        self.in_channels = args.in_channels
        self.hidden_channels = args.hidden_channels
        self.mid_channels = args.mid_channels
        self.out_channels = args.out_channels
        self.dropout_ratio = args.dropout_ratio
        self.gnn_type = args.gnn_type
        self.JK = args.JK
        self.eps = args.eps
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.lambda_cl = args.lambda_cl
        self.contrastive = args.contrastive

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # benign prototype
        self.v_prototype = nn.parameter.Parameter(torch.zeros(1, self.hidden_channels), requires_grad = True)
        self.v_prototype_mal = nn.parameter.Parameter(torch.zeros(1, self.hidden_channels), requires_grad = True)
        nn.init.xavier_uniform_(self.v_prototype)
        nn.init.xavier_uniform_(self.v_prototype_mal)

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
        self.graph_pred_linear = nn.Linear(graph_channels * 2, self.hidden_channels)
        
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
        # node representation
        node_x = self.get_node_rep(x, edge_index)

        # graph representation
        graph_x = torch.cat((global_mean_pool(node_x, batch), global_max_pool(node_x, batch)), dim=1)

        graph_rep = self.graph_pred_linear(graph_x)
        
        # compute 
        features = nn.functional.normalize(graph_rep, dim=1, p=2)
        prototype_benign = self.v_prototype/self.v_prototype.norm(dim=1, p=2)
        prototype_mal = self.v_prototype_mal/self.v_prototype_mal.norm(dim=1, p=2)

        cosine_sim_benign = F.cosine_similarity(features, prototype_benign, dim=1)
        cosine_sim_mal = F.cosine_similarity(features, prototype_mal, dim=1)

        cosine_sim_benign[y==1] = torch.square(cosine_sim_benign[y==1])
        cosine_sim_benign[y==0] = torch.square(1-cosine_sim_benign[y==0])
        cosine_sim_mal[y==0] = torch.square(cosine_sim_mal[y==0])
        cosine_sim_mal[y==1] = torch.square(1-cosine_sim_mal[y==1])

        cl_loss = torch.sum(cosine_sim_benign) + torch.sum(cosine_sim_mal)

        return cl_loss

        #return logits, pred_loss, cl_loss
    
    def evaluate(self, x, edge_index, batch):
        # node representation
        node_x = self.get_node_rep(x, edge_index)

        # graph representation
        graph_x = torch.cat((global_mean_pool(node_x, batch), global_max_pool(node_x, batch)), dim=1)

        # graph classification
        graph_rep = self.graph_pred_linear(graph_x)
        
         # calculate the distance between the sample and the prototype
        features = nn.functional.normalize(graph_rep, dim=1, p=2)
        prototype_benign = F.normalize(self.v_prototype, dim=1, p=2)
        prototype_mal = F.normalize(self.v_prototype_mal, dim=1, p=2)
        cosine_sim_benign = F.cosine_similarity(features, prototype_benign, dim=1)
        cosine_sim_mal = F.cosine_similarity(features, prototype_mal, dim=1)
    
        logits_cl = torch.stack([cosine_sim_benign, cosine_sim_mal], dim=1)

        mask = cosine_sim_benign > cosine_sim_mal
        predicted_labels = torch.logical_not(mask).int()

        return predicted_labels, logits_cl, features
    