import random
import os
import torch

from collections import defaultdict
import sys
sys.path.append('/storage_fast/jnzheng/GCL_security/model')
from Utils.helper import ensure_dir
from sklearn.metrics import roc_auc_score

def save_results(ret: dict, output_path: str):
    """
    Save the results of the model.
    :param ret: a dictionary including the results.
    :param output_path: the path of the output file
    """
    f1 = ret['f1']
    precision = ret['precision']
    recall = ret['recall']
    accuracy = ret['accuracy']
    auc = ret['auc']
    tpr = ret['tpr']
    fpr = ret['fpr']
    ensure_dir(output_path)
    
    # Select different mode to write the results
    if not os.path.exists(output_path):
        with open(output_path, 'w') as f:
            f.write('F-Score Precision Recall Accuracy AUC TPR FPR\n')
            f.write('%.4f %.4f %.4f %.4f %.4f %.4f %.4f\n' % (f1, precision, recall, accuracy, auc, tpr, fpr))
    else:
        with open(output_path, 'a') as f:
            f.write('%.4f %.4f %.4f %.4f %.4f %.4f %.4f\n' % (f1, precision, recall, accuracy, auc, tpr, fpr))

def random_features(vectors: list, labels: list):
    """
    Randomize the features.
    :param vectors: a list of vectors.
    :param labels: a list of labels.
    :return: a list of randomized vectors and a list of randomized labels.
    """
    Vec_Lab = []
    for i in range(len(vectors)):
        vec = vectors[i]
        lab = labels[i]
        vec.append(lab)
        Vec_Lab.append(vec)
    
    random.shuffle(Vec_Lab)

    return [m[:-1] for m in Vec_Lab], [m[-1] for m in Vec_Lab]

def get_device(dev=None):
    """ get device """
    if dev == -1:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        if dev is None:
            # default: use GPU 0
            dev = 0
        device = torch.device(dev)
    else:
        device = torch.device('cpu')

    return device

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.5 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def calculate_f1(precision, recall, beta):
    """ calculate f1 score """
    try:
        return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    except ZeroDivisionError:
        return 0.

def metric2scores(TP, FP, TN, FN):
    correct = TP + TN
    total = correct + FP + FN
    precision = TP / (TP + FP) if (TP + FP)!=0 else 0
    recall = TP / (TP + FN) if (TP + FN)!=0 else 0
    accuracy = correct / total

    f1 = calculate_f1(precision, recall, 1)

    return precision, recall, f1, accuracy

def eval_metrics(y_true, y_pred):
    ret = defaultdict(lambda: "Not present")
    ret['auc'] = roc_auc_score(y_true=y_true, y_score=y_pred)

    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            if y_true[i]:
                TP += 1
            else:
                TN += 1
        else:
            if y_true[i]:
                FN += 1
            else:
                FP += 1

    TPR = TP / (TP + FN) if (TP + FN)!=0 else 0
    FPR = FP / (FP + TN) if (FP + TN)!=0 else 0
    ret['TP'], ret['TN'], ret['FP'], ret['FN'] = TP, TN, FP, FN
    ret['precision'], ret['recall'], ret['f1'], ret['accuracy'] = metric2scores(TP, FP, TN, FN)
    ret['tpr'], ret['fpr'] = TPR, FPR

    return ret

def val_none():
    ret = defaultdict(lambda: "Not present")
    ret['TP'], ret['TN'], ret['FP'], ret['FN'] = 0, 0, 0, 0
    ret['precision'], ret['recall'], ret['f1'], ret['accuracy'] = 0, 0, 0, 0
    ret['auc'] = 0
    return ret
