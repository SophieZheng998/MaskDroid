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
#from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix


class MsDroid_RUN(AbstractRUN):

    def __init__(self, args) -> None:
        super().__init__(args)
        self.contrastive = args.contrastive
        self.seed = args.seed

    def train_one_epoch(self, epoch):
        
        #train_data = self.data.construct_dataset(mode = "train")
        train_idx = list(range(self.data.n_train))

        train_loader = DataLoader(train_idx, batch_size=self.batch_size, shuffle=True, worker_init_fn=np.random.seed(self.seed))
        
        self.model.train()

        running_loss, running_pred_loss, num_batches = 0, 0, 0, 

        pbar = tqdm(enumerate(train_loader), mininterval=2, total = len(train_loader))

        
        for idx, data in pbar:

            # print(data)
            # access data 
            train_subgraphs, _ = self.data.construct_dataset_msdroid(data, "train")
            train_subgraphs = Batch.from_data_list(train_subgraphs)
            train_subgraphs.cuda(self.device)
            
            self.optimizer.zero_grad()
            
            _, pred_loss = self.model(train_subgraphs.x, train_subgraphs.edge_index, train_subgraphs.batch, train_subgraphs.y)
            loss = pred_loss
            loss.backward()
            self.optimizer.step()  
           
            running_loss += loss.detach().item()
            running_pred_loss += pred_loss.detach().item()

            num_batches += 1

        return [running_loss/num_batches]

    def evaluation(self, args, data, model, epoch, base_path, name = "valid"):

        if name == "valid":
            evaluate_idx = list(range(self.data.n_valid))
        elif name == "test":
            evaluate_idx = list(range(self.data.n_test))
        elif name == "train":
            evaluate_idx = list(range(self.data.n_train))

        eval_loader = DataLoader(evaluate_idx, batch_size=self.batch_size, shuffle=False)

        true_labels = []
        predicted_labels = []
        names = []
        label_dictionary = {}

        with torch.no_grad():
            self.model.train()
            for batch in eval_loader:
                evaluate_objects, data_path = self.data.construct_dataset_msdroid(batch, name)
                batch = Batch.from_data_list(evaluate_objects)
                batch.cuda(self.device)
                
                predicted_label, _  = model.evaluate(batch.x, batch.edge_index, batch.batch)
                model.dictionary[data_path[0]] = predicted_label
                true_label = batch.y[0].item()

                predicted_labels.append(predicted_label)
                true_labels.append(true_label)

                label_dictionary[data_path[0]] = predicted_label

                if name == "test" and self.need_record:
                    if true_label == 1 and predicted_label == 1:
                        names.append(data_path[0])

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

        with open(base_path + '{}_dictionary.json'.format(name), 'a') as f:
            json.dump(label_dictionary, f)

            # Check if need to early stop (on validation)
        is_best=False
        early_stop=False

        if name=="valid":
                # TODO
            if recall > data.best_valid_f1:
                data.best_valid_epoch = epoch
                data.best_valid_f1 = recall
                data.patience = 0
                is_best=True
            else:
                data.patience += 1
                if data.patience >= args.patience:
                    print_str = "The best performance epoch is % d " % data.best_valid_epoch
                    print(print_str)
                    early_stop=True

        return is_best, early_stop, n_ret

class MsDroid(torch.nn.Module):

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

        self.dictionary = {}

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

            '''
            if self.contrastive:
                # 加noise
                random_noise = torch.rand_like(x).cuda(self.device) # add noise
                x += torch.sign(x) * F.normalize(random_noise, dim=-1) * self.eps
                if layer == 0:
                    node_x_cl = x
                # end noise
            '''

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

        true_label = y[0].item()

        node_x = self.get_node_rep(x, edge_index)

        # graph representation
        graph_x = torch.cat((global_mean_pool(node_x, batch), global_max_pool(node_x, batch)), dim=1)

        # graph classification
        logits = self.graph_pred_linear(graph_x)

        # probability of being malware
        p_g = F.log_softmax(logits, dim=1)
        # p_g = F.softmax(logits, dim=1)[:, 1]
        # p_g = torch.sigmoid(logits[:, 1])


        # ASL loss
        # pred_loss = torch.negative((1-true_label) * torch.sum(torch.log(1-p_g)) + true_label*torch.min(torch.log(p_g)))
        # pred_loss = torch.negative((1-true_label) * torch.sum(torch.log(1-p_g)) + true_label*torch.min(torch.log(p_g)))
        # if(true_label == 1):
        #     pred_loss = F.nll_loss(p_g, y)
        # else:
        #     scores = []
        #     for j in range(p_g.shape[0]):
        #         scores.append(F.nll_loss(p_g[j], y[j]))
        #     pred_loss = min(scores)
        
        pred_loss = F.nll_loss(p_g, y)
        
        # prediction

        softmax_output = F.softmax(logits, dim=1)
        # max_index = torch.argmax(softmax_output[:, 1])
        # result_logits = softmax_output[max_index]

        # # print(y)
        # print(softmax_output)
        # TODO : min or mean?
        result_logits = softmax_output.min(dim = 0)
        # print(softmax_output.mean(dim = 0))
        # print(true_label)
        # print(result_logits.argmax().item())

        return result_logits, pred_loss

        #return logits, pred_loss, cl_loss
    
    def evaluate(self, x, edge_index, batch):
        # node representation
        node_x = self.get_node_rep(x, edge_index)

        # graph representation
        graph_x = torch.cat((global_mean_pool(node_x, batch), global_max_pool(node_x, batch)), dim=1)

        # graph classification
        logits = self.graph_pred_linear(graph_x)

        softmax_output = F.softmax(logits, dim=1)
        # max_index = torch.argmax(softmax_output[:, 1])
        # result_logits = softmax_output[max_index]
        result_logits = softmax_output.min(dim = 0)
        predicted_label = result_logits.values.argmax().item()

        y_score = softmax_output 

        return predicted_label, y_score
