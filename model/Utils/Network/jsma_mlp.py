#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
This file defines the Jacobian Saliency Map Attack in the paper 'The Limitations of Deep Learning in Adversarial Settings'
[https://arxiv.org/abs/1511.07528v1]
The implementation is based on the paper 'Torchattacks'[https://arxiv.org/pdf/2010.01950.pdf]
"""

import torch
import numpy as np

from logging import Logger

from Utils.Network.helper import get_device

class JSMA():
    def __init__(self, model_1, logger:Logger, model_2=None, substitute_model=None, gamma=0.001, max_iters=0, attack_model="ramda"):
        """
        Jacobian Saliency Map Attack for mlp models (the model to be attacked may be a combination of two models)
        :param model_1: the model_1 to attack
        :param model_2: the model_2 to attack
        :param logger: runtime logging
        :param gamma: highest percentage of perturbed features
        """
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.substitute_model = substitute_model
        self.logger = logger
        self.gamma = gamma
        self.max_iters = max_iters
        self.attack_model = attack_model
        self.device = get_device()
        self.avg_iter = 0

        self.model_1 = self.model_1.to(self.device)
        if self.substitute_model != None:
            # black-box attack
            self.substitute_model = self.substitute_model.to(self.device)
            if self.model_2 != None:
                self.model_2 = self.model_2.to(self.device)
        else:
            # white-box attack
            if self.model_2 != None:
                raise ValueError("Currently, we don't support white-box JSMA attack for two models")

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
            apk = torch.unsqueeze(apk, 0).float()
            if self.label_consistency(apk, target_label):
                continue

            # randomly generating adversarial examples (we do not want to change the original apk)
            random_apk = apk.clone()
            num_features = apk.shape[1]
            random_apk_flatten = random_apk.view(num_features)
            change_num = 0
            while change_num != self.avg_iter:
                perturb_idx = np.random.randint(0, num_features)
                if random_apk_flatten[perturb_idx] == 0:
                    random_apk_flatten[perturb_idx] = 1
                    change_num += 1
            random_apk = random_apk_flatten.view(-1, num_features)

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
        target_label = torch.zeros(1, dtype=torch.long)
        for apk, label in zip(apks, labels):
            # skip benign apks. only generate adversarial examples for malicious apks
            if label == 0:
                continue
            malicious_num += 1

            adversarial_sample, false_negative, iter = self.perturb_single(torch.unsqueeze(apk, 0), target_label)
            
            if adversarial_sample:
                adv_num += 1
                iter_list.append(iter)
            if false_negative:
                fn_num += 1
        
        if adv_num == 0:
            iter_list = [0]

        evasion_rate = adv_num / malicious_num * 100
        self.logger.info("JSMA: Evade success rate: {:.2f}%".format(evasion_rate))
        iter_list = np.array(iter_list)
        self.avg_iter = int(np.average(iter_list)) if iter_list.size != 0 else 0
        max_iter = np.max(iter_list) if iter_list.size != 0 else 0
        min_iter = np.min(iter_list) if iter_list.size != 0 else 0
        median_iter = np.median(iter_list) if iter_list.size != 0 else 0
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
            if self.model_2 != None:
                raise ValueError("Currently, we don't support JSMA attack for two models")
            y_score = self.model_1.score(var_apk)

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
        max_idx = torch.argmax(saliency_map.view(num_features))

        return max_idx

    def perturb_single(self, apk:torch.Tensor, target_label:torch.Tensor):
        # iter indicates the number of features that have been perturbed
        adversarial_sample = False
        false_negative = False
        iter = 0

        # the malicious apk has already been misclassified
        apk = apk.float()
        if self.label_consistency(apk, target_label):
            false_negative = True
            return adversarial_sample, false_negative, iter

        # perturb one feature at a time (cannot perturb two features at the same time due to limited GPU memory)
        num_features = apk.shape[1]
        max_iters = int(np.ceil(num_features * self.gamma)) if self.max_iters == 0 else self.max_iters

        # search space (we only perturb the features that are not 0)
        search_space = torch.eq(apk, 0)
        search_space = search_space.view(num_features)

        # start to generate adversarial examples (we do not want to change the original apk)
        var_apk = apk.clone()
        while (iter < max_iters) and (search_space.sum() != 0):
            # self.logger.info("iter: {} search_space {}".format(iter, search_space.sum()))

            # Jacobian matrix of forward derevative
            jacobian = self.compute_jacobian(var_apk, num_features)

            # computer the saliency map and return the two features that have the highest saliency
            max_idx = self.saliency_map(jacobian, target_label, search_space, num_features)

            # apply modification and build a new apk
            var_apk_flatten = var_apk.view(num_features)
            var_apk_flatten[max_idx] = 1
            var_apk = var_apk_flatten.view(-1, num_features)
            iter += 1

            # update search space
            search_space[max_idx] = 0

            # check if the adversarial example can fool the model
            if self.label_consistency(var_apk, target_label):
                adversarial_sample = True
                break

        return adversarial_sample, false_negative, iter
 
    def label_consistency(self, apk, target_label):
        # check if the prediction is equal to the target label
        y_pred_1 = self.model_1.predict(apk).squeeze().item()
        if self.model_2 != None:
            if self.attack_model == "ramda":
                # model_1 (VAE) provides the encoder for model_2 (MLP)
                mu, sigma, z = self.model_1.get_encoder_output(apk)
                mu_sigma = torch.cat((mu, sigma), dim=1)
                y_pred_2 = self.model_2.predict(mu_sigma).squeeze().item()
                # y_pred_1 is used to identify adversarial examples
                # y_pred_2 is used to detect malware
                # y_pred = y_pred_1 or y_pred_2
                # y_pred = y_pred_1
                y_pred = y_pred_2
            else:
                raise ValueError("attack_model {} is not supported".format(self.attack_model))
        else:
            y_pred = y_pred_1

        return y_pred == target_label