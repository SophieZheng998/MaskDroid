# gnn + proxy
import torch
import numpy as np
from tqdm import tqdm
import warnings
from torch_geometric.data import Batch
warnings.filterwarnings("ignore", category=UserWarning)

from logging import Logger
from torch_geometric.loader import DataLoader
from numpy.random import default_rng
import math
from model.jsma_gnn import JSMA
import matplotlib.pyplot as plt

class JSMA_vae(JSMA):

    def label_consistency(self, apk, target_label):

        adj = torch.sparse.FloatTensor(apk.edge_index, 
                        torch.ones(apk.edge_index.shape[1]).to(self.device), 
                        torch.Size([apk.x.shape[0], apk.x.shape[0]]),).t().to(self.device)
        adj = adj.coalesce()

        y_pred, logits_cl  = self.model.evaluate(apk.x, adj, apk.batch)
        
        
        return y_pred == target_label, [logits_cl]
    
    def compute_jacobian(self, x_features, edge_index, api_batchs, y, steps=10):

        if self.substitute:
            jacobian, search_space = self.compute_jacobian_sub(x_features, edge_index, api_batchs, y, steps)
        else:
            jacobian, search_space = self.compute_jacobian_self(x_features, edge_index, api_batchs, y, steps)

        return jacobian, search_space


    def compute_jacobian_self(self, x_features, edge_index, api_batchs, y, steps=5):
        """Calculate integrated gradient for edges."""

        num_node = x_features.shape[0]
        num_edge = edge_index.shape[1]
        # must perform transpose (.t()) for source-to-target flow
        adj = torch.sparse.FloatTensor(edge_index, 
                                torch.ones(num_edge).to(self.device), 
                                torch.Size([num_node, num_node]),).t().to(self.device)
        adj = adj.coalesce()
        adj.requires_grad = True

        # we only perturb the edges that are not in the original graph
        search_space = torch.eq(adj.to_dense().t(), 0).to(self.device)
        search_space = search_space.contiguous().view(-1)

        # two classes: benign and malicious
        jacobian = torch.zeros([2, num_node, num_node]).to(self.device)

        # adj matrix with all 1s
        baseline_add = torch.ones([num_node, num_node]).to(self.device)

        # integrated gradient (the first new_adj is the original adj matrix)
        scaled_inputs = [baseline_add - (float(k) / steps) * (baseline_add - adj) for k in range(steps, 0, -1)]


        malicious_index = 0        
        for new_adj in scaled_inputs:
            new_adj = new_adj.to_sparse()
            self.model.pretrain_vae.convs[0].cached = True

            _, y_score = self.model.evaluate(x_features, new_adj, api_batchs)
            #y_pred = torch.argmax(y_score, dim=1)

            #malicious_api = torch.where(y_pred == 1)[0]
            #if len(malicious_api) != 0:
            #    malicious_index = malicious_api.tolist()[0]
            #assert malicious_index != -1, "No malicious api found in the apk"

            for i in range(y_score.shape[1]):
                if self.model.pretrain_vae.convs[0]._cached_edge_index[0].grad is not None:
                    self.model.pretrain_vae.convs[0]._cached_edge_index[0].grad.zero_()
                
                used_gradient = y_score[malicious_index][i]
                # change the gradient layout from strided to sparse
                # used_gradient = used_gradient.to_sparse()

                # Note: change torch_geometry's implementation to get the gradient of the edge_index
                # add edge_index.retain_grad() at line 214 (before self._cached_edge_index = (edge_index, edge_weight)) in torch_geometric/nn/conv/gcn_conv.py
                # y_score[malicious_index][i].backward(retain_graph=True)
                used_gradient.backward(retain_graph=True)
                adj_grad = self.model.convs[0]._cached_edge_index[0].grad.t().clone()
                            
                adj_grad_list = adj_grad.detach().to_dense()
                jacobian[i] += adj_grad_list

            self.model.pretrain_vae.convs[0]._cached_edge_index = None
            self.model.pretrain_vae.convs[0].cached = False

        # make impossible perturbations to be negative
        jacobian = jacobian / steps

        return jacobian, search_space
