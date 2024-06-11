#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
This file defines the Jacobian Saliency Map Attack in the paper 'The Limitations of Deep Learning in Adversarial Settings'
[https://arxiv.org/abs/1511.07528v1]
The implementation is based on the paper 'Torchattacks'[https://arxiv.org/pdf/2010.01950.pdf]
"""

import torch
import numpy as np
import math
import os
from pathlib import Path

from logging import Logger

from Utils.Network.helper import get_device

class JSMA():
    def __init__(self, model, logger:Logger, substitute_model=None, gamma=0.001, max_iters=0, attack_model='drebin', matrix_path=None, device = 0):
        """
        Jacobian Saliency Map Attack for conventional machine learning models (SVM, KNN, DT, RF, etc.) 
        :param model: the model to attack
        :param logger: runtime logging
        :param gamma: highest percentage of perturbed features
        :param attack_model: the model to generate adversarial examples (different models use different ways to perturb features)
        """
        super().__init__()
        self.model = model
        self.substitute_model = substitute_model
        self.logger = logger
        self.gamma = gamma
        self.max_iters = max_iters
        self.attack_model = attack_model
        self.device = get_device(device)
        self.avg_iter = 0
        self.new_api_iter = 1

        if self.substitute_model != None:
            self.substitute_model = self.substitute_model.to(self.device)
        else:
            self.model = self.model.to(self.device)

        # perturbation value in each iteration
        if self.attack_model == 'mamadroid' or self.attack_model == 'malscan':
            self.theta = 0.2
        
        if self.attack_model == 'hindroid':
            self.matrix_path = matrix_path
            self.kernel_dir = os.path.join(matrix_path, 'kernels')
            current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            self.hindroid_dir = os.path.join(current_dir.parent.parent, 'Hindroid/Network')
            print(self.hindroid_dir)
            self.test_prediction_path = os.path.join(matrix_path, 'kernels', 'predictions_test')
            self.new_api_iter = 1000

    def setup_kim_features(self, kim_feature_num):
        self.num_merged_features = kim_feature_num[0]
        self.num_string_features = kim_feature_num[1]
        self.num_opcode_features = kim_feature_num[2]
        self.num_api_features = kim_feature_num[3]
        self.num_arm_opcode_features = kim_feature_num[4]
        self.max_iters = self.num_merged_features + self.num_string_features
        self.new_api_iter = 1

    def random_attack(self, apks:torch.Tensor, labels:torch.Tensor):
        malicious_num = 0
        adv_num = 0
        # generate an adversarial example for one malware at a time
        # we hardcode the target label to be 0 (benign apk)
        target_label = torch.zeros(1, dtype=torch.long).to(self.device)
        for apk, label in zip(apks, labels):
            # skip benign apks. only generate adversarial examples for malicious apks
            if label == 0:
                continue
            malicious_num += 1

            # if the malicious apk has already been misclassified, skip this apk
            apk = torch.unsqueeze(apk, 0)
            if self.label_consistency(apk, target_label):
                continue

            # randomly generating adversarial examples (we do not want to change the original apk)
            random_apk = apk.clone()
            num_features = apk.shape[1]
            change_num = 0
            while change_num != self.avg_iter:
                if self.attack_model == 'homdroid':
                    # only perturb the first 426 features (existence of sensitive apis)
                    perturb_idx = np.random.randint(0, 426)
                elif self.attack_model == 'kim':
                    # max_iters defines the number of features to be perturbed
                    perturb_idx = np.random.randint(0, self.max_iters)
                else:
                    perturb_idx = np.random.randint(0, num_features)

                if self.attack_model == 'drebin' or self.attack_model == 'sdac'\
                or self.attack_model == 'homdroid' or self.attack_model == 'hindroid'\
                or self.attack_model == 'kim':
                    random_apk_flatten = random_apk.view(num_features)
                    if random_apk_flatten[perturb_idx] == 0:
                        random_apk_flatten[perturb_idx] = 1
                        change_num += 1

                elif self.attack_model == 'mamadroid':
                    num_abstraction = int(pow(num_features, 0.5))
                    start_idx = math.floor(perturb_idx / num_abstraction) * num_abstraction
                    end_idx = start_idx + num_abstraction
                    idx = perturb_idx % num_abstraction
                    random_apk_flatten = random_apk.view(num_features)
                    
                    # add new api calls and normalize the feature vector
                    selected_feature = random_apk_flatten[start_idx:end_idx]
                    selected_feature[idx] += self.theta
                    total = torch.sum(selected_feature)
                    selected_feature = selected_feature / total
                    random_apk_flatten[start_idx: end_idx] = selected_feature
                    change_num += 1

                elif self.attack_model == 'malscan':
                    random_apk_flatten = random_apk.view(num_features)
                    random_apk_flatten[perturb_idx] += self.theta
                    change_num += 1

                else:
                    raise NotImplementedError("Attack model {} is not implemented".format(self.attack_model))

            # check if the adversarial example can fool the model
            if self.label_consistency(random_apk, target_label):
                adv_num += 1

        self.logger.info("Random: Evade success rate: {:.2f}%".format(adv_num / malicious_num * 100))

    def attack(self, apks, labels):        
        malicious_num = 0
        adv_num = 0
        fn_num = 0
        iter_list = []
        # generate an adversarial example for one malware at a time
        # we hardcode the target label to be 0 (benign apk)
        target_label = torch.zeros(1, dtype=torch.long).to(self.device)
        for i, (apk, label) in enumerate(zip(apks, labels)):
            #self.logger.info("Current apk: {}".format(i))

            # skip benign apks. only generate adversarial examples for malicious apks
            if label == 0:
                continue
            malicious_num += 1

            adversarial_sample, false_negative, iter = self.perturb_single(torch.unsqueeze(apk, 0), target_label)
            
            if false_negative:
                fn_num += 1
                continue
            
            if adversarial_sample:
                adv_num += 1
                iter_list.append(iter)
                self.logger.debug("JSMA: Success! Avg iter: {}".format(iter))
            else:
                self.logger.debug("JSMA: Fail!")
        
        evasion_rate = adv_num / (malicious_num-fn_num) * 100
        self.logger.info("JSMA: Evade success rate: {:.2f}%".format(evasion_rate))

        iter_list = np.array(iter_list) * self.new_api_iter
        self.avg_iter = int(np.average(iter_list)) if len(iter_list) != 0 else 0
        max_iter = np.max(iter_list) if len(iter_list) != 0 else 0
        min_iter = np.min(iter_list) if len(iter_list) != 0 else 0
        median_iter = np.median(iter_list) if len(iter_list) != 0 else 0
        self.logger.info("#Changes: [max, min, avg, median] = [{}, {}, {:.2f}, {}]".format(max_iter, min_iter, self.avg_iter, median_iter))

        fn_rate = fn_num / malicious_num * 100
        self.logger.info("False negative rate: {:.2f}%".format(fn_rate))
        
        total_misclassified_rate = (adv_num + fn_num) / malicious_num * 100
        self.logger.info("Total misclassified rate: {:.2f}%".format(total_misclassified_rate))

    def compute_jacobian(self, apk, num_features):
        var_apk = apk.clone().detach()
        var_apk.requires_grad = True
        if self.substitute_model != None:
            y_score = self.substitute_model.score(var_apk)
        else:
            y_score = self.model.score(var_apk)

        # compute the jacobian matrix
        jacobian = torch.zeros(y_score.shape[1], num_features)
        for i in range(y_score.shape[1]):
            if var_apk.grad is not None:
                var_apk.grad.data.zero_()
            y_score[0, i].backward(retain_graph=True)
            jacobian[i] = var_apk.grad.squeeze().view(-1, num_features).clone()

        return jacobian.to(self.device)

    @torch.no_grad()
    def saliency_map(self, jacobian, target_label, search_space, num_features):
        # The sum of all features' derivative with respect to each class
        all_sum = torch.sum(jacobian, dim=0, keepdim=True)
        # The derivative of the target class
        target_grad = jacobian[target_label]
        # The derivative of the other classes
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

        # get the most significant feature pair
        max_idx = torch.topk(saliency_map.view(num_features), k=self.new_api_iter).indices.cpu().numpy()

        return max_idx

    def perturb_single(self, apk:torch.Tensor, target_label:torch.Tensor):
        # iter indicates the number of features that have been perturbed
        adversarial_sample = False
        false_negative = False
        iter = 0

        # the malicious apk has already been misclassified
        if self.label_consistency(apk, target_label):
            false_negative = True
            return adversarial_sample, false_negative, iter

        # perturb one feature at a time (cannot perturb two features at the same time due to limited GPU memory)
        num_features = apk.shape[1]
        max_iters = int(np.ceil(num_features * self.gamma)) if self.max_iters == 0 else self.max_iters

        # search space
        if self.attack_model == 'drebin' or self.attack_model == 'sdac' or self.attack_model == 'hindroid':
            # in drebin/sdac, we only perturb the features that are not 0
            search_space = torch.eq(apk, 0)
            search_space = search_space.view(num_features)
        elif self.attack_model == 'homdroid':
            # in homdroid, we only perturb the first 426 features that indicate whether a sensitive API is used or not
            search_space = torch.eq(apk, 0)
            search_space = search_space.view(num_features)
            search_space[426:] = False
        elif self.attack_model == 'kim':
            # in kim (multi-model), we only perturb the existing features (merged and string features)
            search_space = torch.eq(apk, 0)
            search_space = search_space.view(num_features)
            search_space[self.num_merged_features + self.num_string_features:] = False
        else:
            search_space = torch.ones(num_features)

        # start to generate adversarial examples (we do not want to change the original apk)
        var_apk = apk.clone()
        while (iter < max_iters) and (search_space.sum() != 0):
            # Jacobian matrix of forward derevative
            jacobian = self.compute_jacobian(var_apk, num_features)

            # computer the saliency map and return the two features that have the highest saliency
            max_idx = self.saliency_map(jacobian, target_label, search_space, num_features)

            # apply modification and build a new apk
            if self.attack_model == 'drebin' or self.attack_model == 'sdac'\
            or self.attack_model == 'homdroid' or self.attack_model == 'hindroid'\
            or self.attack_model == 'kim':
                var_apk_flatten = var_apk.view(num_features)
                # var_apk_flatten[max_idx] must be 0, otherwise it will be chosen
                var_apk_flatten[max_idx] = 1
                var_apk = var_apk_flatten.view(-1, num_features)
                # update search space
                search_space[max_idx] = 0

            elif self.attack_model == 'mamadroid':
                # retrieve the feature vector to be modified
                num_abstraction = int(pow(num_features, 0.5))
                start_idx = math.floor(max_idx / num_abstraction) * num_abstraction
                end_idx = start_idx + num_abstraction
                idx = max_idx % num_abstraction
                var_apk_flatten = var_apk.view(num_features)

                # add new api calls and normalize the feature vector
                selected_features = var_apk_flatten[start_idx: end_idx]
                selected_features[idx] += self.theta
                total = torch.sum(selected_features)
                selected_features = selected_features / total
                var_apk_flatten[start_idx: end_idx] = selected_features

            elif self.attack_model == 'malscan':
                var_apk_flatten = var_apk.view(num_features)
                var_apk_flatten[max_idx] += self.theta
            else:
                raise NotImplementedError("Attack model {} is not implemented".format(self.attack_model))
            iter += 1

            # check if the adversarial example can fool the model
            if self.label_consistency(var_apk, target_label):
                adversarial_sample = True
                break

        return adversarial_sample, false_negative, iter
 
    def label_consistency(self, apk, target_label):
        #if self.attack_model == 'hindroid':
        #    from Hindroid.Network.generate_kernels import get_kernels, save_kernel, save_label
        #    import subprocess
        #    test_kernels = get_kernels(apk, self.substitute_model.matrix_B, self.substitute_model.matrix_P, self.substitute_model.matrix_I, self.substitute_model.matrix_A)
        #    for kernel_name, kernel in test_kernels.items():
        #        save_kernel(self.kernel_dir, kernel_name, kernel, 'test')
        #    save_label(self.kernel_dir, [1], 'test')
        #    c_command_test = [f'{self.hindroid_dir}/Mkl/svm-predict', f'{self.matrix_path}/kernels/y_test', f'{self.matrix_path}/kernels/model_file', f'{self.matrix_path}/kernels/predictions_test']
        #    _ = subprocess.run(c_command_test, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #    with open(self.test_prediction_path, 'r') as f:
        #        test_predictions = f.read().splitlines()
        #        test_predictions = [int(x) for x in test_predictions]
        #    assert len(test_predictions) == 1
        #    return test_predictions[0] == target_label
        
        if self.attack_model == 'kim':
            merged_feature = apk[:, :self.num_merged_features]
            string_feature = apk[:, self.num_merged_features:self.num_merged_features + self.num_string_features]
            opcode_feature = apk[:, self.num_merged_features + self.num_string_features:self.num_merged_features + self.num_string_features + self.num_opcode_features]
            api_feature = apk[:, self.num_merged_features + self.num_string_features + self.num_opcode_features:self.num_merged_features + self.num_string_features + self.num_opcode_features + self.num_api_features]
            arm_opcode_feature = apk[:, self.num_merged_features + self.num_string_features + self.num_opcode_features + self.num_api_features:]
            kim_apk = [merged_feature, string_feature, opcode_feature, api_feature, arm_opcode_feature]
            y_pred = self.model.predict(kim_apk).squeeze().item()
        else:
            # check if the prediction is equal to the target label
            if self.substitute_model != None:
                # e.g., model == svm
                apk_numpy = apk.cpu().numpy()
                y_pred_numpy = self.model.predict(apk_numpy)
                y_pred = torch.from_numpy(y_pred_numpy).to(self.device).squeeze().item()
            else:
                # white-box attacks on drebin
                y_pred = self.model.predict(apk).squeeze().item()
        
        return y_pred == target_label
