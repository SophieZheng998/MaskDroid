# gnn + mlp
import torch
import numpy as np
from tqdm import tqdm
import warnings
import math
import pandas as pd
import os
import json
from torch_geometric.data import Batch
warnings.filterwarnings("ignore", category=UserWarning)

from logging import Logger
from torch_geometric.loader import DataLoader
from numpy.random import default_rng

#from Utils.Network.helper import get_device
#from Utils.Network.gnn_pyq import convert_real_batch

class JSMA():
    def __init__(self, model, args, logger:Logger, steps=10, substitute_model = None):
        super().__init__()
        self.model = model#.half()
        self.substitute_model = substitute_model#.half()
        self.logger = logger
        self.steps = steps
        self.device = torch.device(args.cuda)
        self.max_iters = 100
        self.num_new_edge_iter = 5
        # 155 for gcn
        self.avg_iter = 100
        self.ablation = args.ablation
        if self.ablation:
            self.root = "../data/ASR/filenames"
        else:
            self.root = "../weights"

        with open(os.path.join(self.root, "attack.json"), "r") as f:
            self.files = json.load(f)
        self.n_files = len(self.files)
        self.model = self.model.to(self.device)
        self.orders = []

        self.substitute = False

        if self.substitute_model != None:
            # back-box attack
            self.substitute_model = self.substitute_model.to(self.device)
            self.substitute = True
        else:
            # white-box attack
            self.substitute_model = self.model

        self.logits_benign_list = []

    def construct_dataset(self, indices):
        
        dataset = []
        
        for idx in indices:
            data_path = self.files[idx]
            self.orders.append(data_path)
            data_object = torch.load(data_path)            
            dataset.append(data_object)
        
        return dataset

    def attack(self):

        loader = DataLoader(list(range(self.n_files)), batch_size=1, shuffle=False)

        malicious_num = 0
        adv_num = 0
        fn_num = 0
        skip_num = 0
        iter_list = []
        total_edge_list = []
        # generate an adversarial example for one malware at a time
        # we hardcode the target label to be 0 (benign apk)
        target_label = torch.zeros(1, dtype=torch.long).to(self.device)
        pbar = tqdm(enumerate(loader), mininterval=2, total = len(loader))

        #feature_dict = {}

        for i, apk in pbar:

            # TODO
            #if i < 147:
            #    continue

            test_objects = self.construct_dataset(apk)

            apk = Batch.from_data_list(test_objects)
            
            #apk, _ = convert_real_batch(apk)
            apk.to(self.device)
            #apk.x = torch.tensor(apk.x, dtype=torch.float16)
            
            self.logger.info("Current apk: {} num_edges: {}".format(i, apk.edge_index.shape[1]))

            # labels for all apis (subgraphs) in an apk
            api_labels = apk.y

            # skip benign apks (if one of its apis is benign, the apk is benign)
            if api_labels[0] == 0:
                continue
            malicious_num += 1

            original_result, original_logits = self.label_consistency(apk, target_label)

            original_logits = [tensor.cpu().detach().numpy()[0] for tensor in original_logits]

            if original_result:
                false_negative = True
                fn_num += 1
                self.logger.debug("JSMA: False Negative!, logits: {}".format(original_logits))
                continue

            adversarial_sample, iter, graph_too_large, logits_benign = self.perturb_single(apk, target_label)
            
            if adversarial_sample:
                adv_num += 1
                iter_list.append(iter)
                total_edge_list.append(apk.edge_index.shape[1])
                self.logits_benign_list.append(logits_benign)
                output_logits = [tensor.cpu().detach().numpy()[0] for tensor in logits_benign]
                self.logger.debug("JSMA: Success! Avg iter: {}, original logits: {}, logits after attack: {}".format(iter * self.num_new_edge_iter, original_logits, output_logits))
                self.logger.debug("The total number of edges of this apk : {}".format(apk.edge_index.shape[1]))
            else:
                if graph_too_large:
                    self.logger.debug("JSMA: Graph too large! ()")
                    skip_num += 1
                else:
                    output_logits = [tensor.cpu().detach().numpy()[0] for tensor in logits_benign]
                    self.logger.debug("JSMA: Fail!, original logits:{}, logits after attack: {}".format(original_logits, output_logits))

            
            #feature_dict[i] = (feature.detach().cpu().numpy(), adversarial_sample)
            #y_pred, logits_cl,  features  = self.model.evaluate(apk.x, adj, apk.batch)
            
        if adv_num == 0:
            iter_list = [0]
        
        #np.save("mask_droid_adversarial_gnn_new.npy", feature_dict)

        self.logger.info("Total number of malicious apks: {}".format(malicious_num))

        skip_rate = skip_num / malicious_num * 100 if malicious_num != 0 else 0
        self.logger.info("JSMA: Skip rate (graph is too large to process): {:.2f}%".format(skip_rate))

        evasion_rate = adv_num / malicious_num * 100 if malicious_num != 0 else 0
        self.logger.info("JSMA: Evade success rate: {:.2f}%".format(evasion_rate))

        avg_iter = np.average(iter_list) if iter_list else 0
        APR = (np.sum(iter_list)*self.num_new_edge_iter) / np.sum(total_edge_list) if iter_list else 0 
        self.logger.debug("JSMA: Avg iter: {}".format(avg_iter))
        self.logger.debug("Modify {} edges on average".format(avg_iter * self.num_new_edge_iter))
        self.logger.debug("APR:  {:.2f}%".format(APR * 100))

        self.logger.debug("# edges modified : {}".format(np.sum(iter_list)*self.num_new_edge_iter))
        self.logger.debug("# edges : {}".format(np.sum(total_edge_list)))

        iter_list = np.array(iter_list) if iter_list else np.array([0])
        self.avg_iter = int(np.average(iter_list)) if iter_list.size != 0 else 0
        max_iter = np.max(iter_list) if iter_list.size != 0 else 0
        min_iter = np.min(iter_list) if iter_list.size != 0 else 0
        median_iter = np.median(iter_list) if iter_list.size != 0 else 0
        self.logger.info("#Changes: [max, min, avg, median] = [{}, {}, {:.2f}, {}]".format(max_iter, min_iter, self.avg_iter, median_iter))
        
        fn_rate = fn_num / malicious_num * 100
        self.logger.info("False negative rate: {:.2f}%".format(fn_rate))
        
        total_misclassified_rate = (adv_num + fn_num) / malicious_num * 100 if malicious_num != 0 else 0
        self.logger.info("Total misclassified rate: {:.2f}%".format(total_misclassified_rate))
        return evasion_rate, self.avg_iter, total_misclassified_rate

    def perturb_single(self, apk, target_label):
        var_apk = apk.clone()

        # iter indicates the number of features that have been perturbed
        adversarial_sample = False
        iter = 0
        logits_benign = []

        # Todo: handle the case when the graph is too large)
        graph_too_large = False

        # number of features is the number of edges in the graph
        num_node = var_apk.x.shape[0]
        num_features = num_node * num_node
        if num_features > 40000000:
            graph_too_large = True
            return adversarial_sample, iter, graph_too_large, logits_benign

        while iter < self.max_iters:
            # Jacobian matrix of forward derevative
            integrated_grad_list, search_space = self.compute_jacobian(var_apk.x, var_apk.edge_index, var_apk.batch, var_apk.y, self.steps)

            # computer the saliency map and return the two features that have the highest saliency
            n1, n2 = self.saliency_map(integrated_grad_list, target_label, num_features, search_space)

            # apply modification and build a new apk
            new_edge = torch.tensor([n1,n2]).to(self.device)
            var_apk.edge_index = torch.cat([var_apk.edge_index, new_edge], dim=-1)
            iter += 1

            # check if the adversarial example can fool the model
            result, logits_benign, =  self.label_consistency(var_apk, target_label)
            if result:
                adversarial_sample = True
                break


        return adversarial_sample, iter, graph_too_large, logits_benign, 

    @torch.no_grad()
    def saliency_map(self, jacobian, target_label, num_features, search_space):

        jacobian = jacobian.view(-1, num_features)

        # The sum of all features' derivative with respect to each class
        all_sum = torch.sum(jacobian, dim=0, keepdim=True)
        # The derivative of the target class
        target_grad = jacobian[target_label]
        # The derivative of the other classes (initially designed for multi-class classification)
        other_grad = all_sum - target_grad
        
        # blank out the features that are not in the search space
        increase_coef = torch.eq(search_space, 0).float().to(self.device)

         # The sum of the forward derivative of any feature 
        target_tmp = target_grad.clone()
        target_tmp -= increase_coef * torch.max(torch.abs(target_grad))
        alpha = target_tmp.view(-1, num_features)

        # The sum of the other derivative of any 2 features 
        other_tmp = other_grad.clone()
        other_tmp += increase_coef * torch.max(torch.abs(other_grad))
        beta = other_tmp.view(-1, num_features)

        # According to the definition of saliency map in the paper (formulas 8 and 9),
        # those elements in the saliency map that doesn't satisfy the requirement will be blanked out.
        mask1 = torch.gt(alpha, 0.0)
        mask2 = torch.lt(beta, 0.0)
        mask = torch.mul(mask1, mask2)

        # Do the multiplication according to formula 10 in the paper
        saliency_map = torch.mul(
            torch.mul(alpha, torch.abs(beta)), mask.float())

        # get the most significant features (the top 50 features with the highest saliency)
        max_idx = torch.topk(saliency_map.view(num_features), k=self.num_new_edge_iter).indices.cpu().numpy()

        num_node = int(pow(num_features, 0.5))
        n1 = max_idx // num_node
        n2 = max_idx % num_node

        return n1, n2

    def compute_jacobian(self, x_features, edge_index, api_batchs, y, steps=10):
        """Calculate integrated gradient for edges."""

        num_node = x_features.shape[0]
        num_edge = edge_index.shape[1]
        # must perform transpose (.t()) for source-to-target flow
        adj = torch.sparse.FloatTensor(edge_index, 
                                torch.ones(num_edge).to(self.device), 
                                torch.Size([num_node, num_node]),).t().to(self.device)
        adj = adj.coalesce()
        #adj = adj.detach()
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
            #new_adj = new_adj.detach()
            self.substitute_model.convs[0].cached = True
            y_score, _ = self.substitute_model(x_features, new_adj, api_batchs, y)
            y_pred = torch.argmax(y_score, dim=1)

            #malicious_api = torch.where(y_pred == 1)[0]
            #if len(malicious_api) != 0:
            #    malicious_index = malicious_api.tolist()[0]
            #assert malicious_index != -1, "No malicious api found in the apk"

            for i in range(y_score.shape[1]):
                if self.substitute_model.convs[0]._cached_edge_index[0].grad is not None:
                    self.substitute_model.convs[0]._cached_edge_index[0].grad.zero_()
                
                used_gradient = y_score[malicious_index][i]
                # change the gradient layout from strided to sparse
                # used_gradient = used_gradient.to_sparse()

                # Note: change torch_geometry's implementation to get the gradient of the edge_index
                # add edge_index.retain_grad() at line 214 (before self._cached_edge_index = (edge_index, edge_weight)) in torch_geometric/nn/conv/gcn_conv.py
                # y_score[malicious_index][i].backward(retain_graph=True)
                used_gradient.backward(retain_graph=True)
                adj_grad = self.substitute_model.convs[0]._cached_edge_index[0].grad.t().clone()
                            
                adj_grad_list = adj_grad.detach().to_dense()
                jacobian[i] += adj_grad_list

            self.substitute_model.convs[0]._cached_edge_index = None
            self.substitute_model.convs[0].cached = False

        # make impossible perturbations to be negative
        jacobian = jacobian / steps

        return jacobian, search_space

    def label_consistency(self, apk, target_label):

        adj = torch.sparse.FloatTensor(apk.edge_index, 
                        torch.ones(apk.edge_index.shape[1]).to(self.device), 
                        torch.Size([apk.x.shape[0], apk.x.shape[0]]),).t().to(self.device)
        adj = adj.coalesce()


        y_pred, logits, feature = self.model.evaluate(apk.x, adj, apk.batch)

        #logits, _,  = self.model(apk.x, adj, apk.batch, apk.y)
        #y_pred = logits.argmax(dim=1)
        y_pred = torch.max(y_pred)
        
        return y_pred == target_label, [logits]

    def compute_jacobian_sub(self, x_features, edge_index, api_batchs, y, steps=10):
        """Calculate integrated gradient for edges."""

        num_node = x_features.shape[0]
        num_edge = edge_index.shape[1]
        # must perform transpose (.t()) for source-to-target flow
        adj = torch.sparse.FloatTensor(edge_index, 
                                torch.ones(num_edge).to(self.device), 
                                torch.Size([num_node, num_node]),).t().to(self.device)
        adj = adj.coalesce()
        #adj = adj.half()
        #adj = adj.to(torch.float16)
        #adj = adj.detach()
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
            #new_adj = new_adj.half()
            #new_adj = new_adj.to(torch.float16)
            #new_adj = new_adj.detach()
            #new_adj.requires_grad = True
            self.substitute_model.convs[0].cached = True
            y_score, _ = self.substitute_model(x_features, new_adj, api_batchs, y)
    

            for i in range(y_score.shape[1]):
                if self.substitute_model.convs[0]._cached_edge_index[0].grad is not None:
                    self.substitute_model.convs[0]._cached_edge_index[0].grad.zero_()
            
                used_gradient = y_score[malicious_index][i]
                # change the gradient layout from strided to sparse
                # used_gradient = used_gradient.to_sparse()

                # Note: change torch_geometry's implementation to get the gradient of the edge_index
                # add edge_index.retain_grad() at line 214 (before self._cached_edge_index = (edge_index, edge_weight)) in torch_geometric/nn/conv/gcn_conv.py
                # y_score[malicious_index][i].backward(retain_graph=True)
                used_gradient.backward(retain_graph=True)
                adj_grad = self.substitute_model.convs[0]._cached_edge_index[0].grad.t().clone()
                            
                adj_grad_list = adj_grad.detach().to_dense()
                jacobian[i] += adj_grad_list

            self.substitute_model.convs[0]._cached_edge_index = None
            self.substitute_model.convs[0].cached = False

        # make impossible perturbations to be negative
        jacobian = jacobian / steps

        return jacobian, search_space

'''
 
    def random_attack(self):
        loader = DataLoader(list(range(self.n_files)), batch_size=1, shuffle=False)
        malicious_num = 0
        adv_num = 0
        fn_num = 0

        avg_new_edge = self.num_new_edge_iter * self.avg_iter
        # generate an adversarial example for one malware at a time
        # we hardcode the target label to be 0 (benign apk)
        target_label = torch.zeros(1, dtype=torch.long).to(self.device)
        pbar = tqdm(enumerate(loader), mininterval=2, total = len(loader))

        for _, apk in pbar:

            test_objects = self.construct_dataset(apk)

            apk = Batch.from_data_list(test_objects)
            
            #apk, _ = convert_real_batch(apk)
            apk.to(self.device)

            # labels for all apis (subgraphs) in an apk
            api_labels = apk.y

            # skip benign apks (if one of its apis is benign, the apk is benign)
            if api_labels[0] == 0:
                continue
            malicious_num += 1

            # if the malicious apk has already been misclassified, skip this apk
            result, _ = self.label_consistency(apk, target_label)
            if result:
                fn_num +=  1
                continue
            
            # randomly generating adversarial examples (we do not want to change the original apk)
            random_apk = apk.clone()
            num_node = random_apk.x.shape[0]
            num_features = num_node * num_node
            rng = default_rng()
            avg = min(num_features, avg_new_edge)
            max_idx = rng.choice(num_features, avg, replace=False)
            num_node = int(pow(num_features, 0.5))
            n1 = max_idx // num_node
            n2 = max_idx % num_node


            # apply modification and build a new apk
            new_edge = torch.tensor([n1,n2]).to(self.device)
            random_apk.edge_index = torch.cat([random_apk.edge_index, new_edge], dim=-1)

            # check if the adversarial example can fool the model
            result, _ = self.label_consistency(random_apk, target_label)

            if result:
                adv_num += 1
        
        self.logger.info("Random: Avg new edges: {}".format(avg_new_edge))
        self.logger.info("Random: Evade success rate: {:.2f}%".format(adv_num / malicious_num * 100))
        self.logger.info("False negative rate: {:.2f}%".format(fn_num / malicious_num * 100))
        total_misclassified_rate = (adv_num + fn_num) / malicious_num * 100 if malicious_num != 0 else 0
        self.logger.info("Total misclassified rate: {:.2f}%".format(total_misclassified_rate))

        df = pd.DataFrame(data = {"file_name":self.orders})
        df.to_csv("file_names.csv", index = False)

    def SAI_attack(self):
        loader = DataLoader(list(range(self.n_files)), batch_size=1, shuffle=False)
        malicious_num = 0
        adv_num = 0
        fn_num = 0
        iter_list = []

        #avg_new_edge = self.num_new_edge_iter * self.avg_iter
        # generate an adversarial example for one malware at a time
        # we hardcode the target label to be 0 (benign apk)
        target_label = torch.zeros(1, dtype=torch.long).to(self.device)
        pbar = tqdm(enumerate(loader), mininterval=2, total = len(loader))

        for _, apk in pbar:

            test_objects = self.construct_dataset(apk)

            apk = Batch.from_data_list(test_objects)
            
            #apk, _ = convert_real_batch(apk)
            apk.to(self.device)

            # labels for all apis (subgraphs) in an apk
            api_labels = apk.y

            # skip benign apks (if one of its apis is benign, the apk is benign)
            if api_labels[0] == 0:
                continue
            malicious_num += 1

            # start from here, do the attack
            
            
            adversarial_sample, false_negative, iter, new_logits_result = self.SAI_add(apk, target_label)

            if false_negative:
                fn_num += 1
                self.logger.debug("JSMA: False Negative!")
                continue

            if adversarial_sample:
                adv_num += 1
                iter_list.append(iter)
                output_logits = [tensor.cpu().detach().numpy()[0] for tensor in new_logits_result]
                self.logger.debug("JSMA: Success! Avg iter: {}, logits of benign: {}".format(iter * self.num_new_edge_iter, output_logits))
            else:
                self.logger.debug("JSMA: Fail!")


        if adv_num == 0:
            iter_list = [0]

        self.logger.info("Total number of malicious apks: {}".format(malicious_num))
        evasion_rate = adv_num / malicious_num * 100 if malicious_num != 0 else 0
        self.logger.info("JSMA: Evade success rate: {:.2f}%".format(evasion_rate))
        avg_iter = np.average(iter_list) if iter_list else 0
        self.logger.debug("JSMA: Avg iter: {}".format(avg_iter))
        self.logger.debug("Modify {} edges on average".format(avg_iter * self.num_new_edge_iter))
        iter_list = np.array(iter_list) if iter_list else np.array([0])
        self.avg_iter = int(np.average(iter_list)) if iter_list.size != 0 else 0
        max_iter = np.max(iter_list) if iter_list.size != 0 else 0
        min_iter = np.min(iter_list) if iter_list.size != 0 else 0
        median_iter = np.median(iter_list) if iter_list.size != 0 else 0
        self.logger.info("#Changes: [max, min, avg, median] = [{}, {}, {:.2f}, {}]".format(max_iter, min_iter, self.avg_iter, median_iter))
        fn_rate = fn_num / malicious_num * 100
        self.logger.info("False negative rate: {:.2f}%".format(fn_rate))
        total_misclassified_rate = (adv_num + fn_num) / malicious_num * 100 if malicious_num != 0 else 0
        self.logger.info("Total misclassified rate: {:.2f}%".format(total_misclassified_rate))
        return evasion_rate, self.avg_iter, total_misclassified_rate

    def SAI_add(self, apk, target_label):

        var_apk = apk.clone()
        adversarial_sample = False
        false_negative = False
        iter = 0

        # if false negative
        result, original_logits_result = self.label_consistency(var_apk, target_label)
        if result:
            false_negative = True
            return adversarial_sample, false_negative, iter,  original_logits_result
        
        # if the graph is too large
        num_node = var_apk.x.shape[0]
        num_features = num_node * num_node
        #if num_features > 1000000:
        #    graph_too_large = True
        #    return adversarial_sample, false_negative, iter, graph_too_large, original_logits_result
        
        num_node = var_apk.x.shape[0]
        num_features = num_node * num_node
        rng = default_rng()
        avg = min(num_features, self.num_new_edge_iter)
        
        while iter < self.max_iters:
            
            result, original_logits_result = self.label_consistency(var_apk, target_label)
            temp_store_edge = var_apk.edge_index
            
            max_idx = rng.choice(num_features, avg, replace=False)
            num_node = int(pow(num_features, 0.5))
            n1 = max_idx // num_node
            n2 = max_idx % num_node

            # apply modification and build a new apk
            new_edge = torch.tensor([n1,n2]).to(self.device)
            var_apk.edge_index = torch.cat([var_apk.edge_index, new_edge], dim=-1)

            result, new_logits_result = self.label_consistency(var_apk, target_label)
            iter += 1

            # if the sample can already fool the model, return 
            if result:
                adversarial_sample = True
                break

            # check if the probability of predicting the target class increases
            if len(new_logits_result)==1:
                output_logits_before = original_logits_result[0][0]
                output_logits = new_logits_result[0][0]
                benign_before = math.exp(output_logits_before[0].item())/(math.exp(output_logits_before[0].item()) + math.exp(output_logits_before[1].item()))
                benign_after = math.exp(output_logits[0].item())/(math.exp(output_logits[0].item()) + math.exp(output_logits[1].item()))
                
                # if no increment
                if benign_after <= benign_before:
                    var_apk.edge_index = temp_store_edge

            else:
                logits_cl_before = original_logits_result[0][0]
                logits_cl = new_logits_result[0][0]
                logits_mlp_before = original_logits_result[1][0]
                logits_mlp = new_logits_result[1][0]

                benign_cl_before = math.exp(logits_cl_before[0].item())/(math.exp(logits_cl_before[0].item()) + math.exp(logits_cl_before[1].item()))
                benign_mlp_before = math.exp(logits_mlp_before[0].item())/(math.exp(logits_mlp_before[0].item()) + math.exp(logits_mlp_before[1].item()))
                benign_cl = math.exp(logits_cl[0].item())/(math.exp(logits_cl[0].item()) + math.exp(logits_cl[1].item()))
                benign_mlp = math.exp(logits_mlp[0].item())/(math.exp(logits_mlp[0].item()) + math.exp(logits_mlp[1].item()))
                
                if (benign_cl <= benign_cl_before) or (benign_mlp <= benign_mlp_before):
                    var_apk.edge_index = temp_store_edge
                
            
        return adversarial_sample, false_negative, iter, new_logits_result
'''
