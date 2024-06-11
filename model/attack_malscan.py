import torch
import os
import numpy as np
import joblib
import csv
from logging import Logger
import random
from itertools import islice
from Utils.Network.mlp import MLP, mlp_train, mlp_evaluate
from Utils.Network.jsma_cml import JSMA
from Utils.Network.data import SimpleDataset
from Utils.Network.helper import get_device, eval_metrics
from Mamadroid import Mamadroid
import sys
from parse import parse_args
import logging
import json


def ensureDir(dir_path):
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)

def obtain_feature_paths(feature_dirs: list, centrality_type: str):
    """
    Obtain the feature paths.
    :param feature_dirs: a list of feature directories.
    :param centrality_type: the type of centrality.
    :return: a list of feature file paths.
    """
    feature_file_paths = []
    for feature_dir in feature_dirs:
        feature_file_path = os.path.join(feature_dir, centrality_type + '_features.csv')
        feature_file_paths.append(feature_file_path)

    return feature_file_paths

def degree_centrality_feature(feature_dirs: list, train_flag: str, dic: dict):
    """
    Query the degree centrality feature.
    :param feature_dirs: a list of feature directories.
    :param train_flag: the flag of training or not.
    :param dic: a dictionary including apk names.
    """
    feature_file_paths = obtain_feature_paths(feature_dirs, 'degree')
    return query_centrality_features(feature_file_paths, train_flag, dic)

def query_centrality_features(feature_file_paths: list, train_flag: str, dic: dict):
    """
    Query the centrality feature.
    :param feature_file_paths: a list of feature file paths.
    :param train_flag: the flag of training or not.
    :param dic: a dictionary including apk names.
    """
    if train_flag == 'train':
        train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels  = read_train_data(feature_file_paths, dic)
        return train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels
    elif train_flag == 'test':
        vectors, labels = read_test_data(feature_file_paths, dic)
        return vectors, labels
    else:
        raise ValueError("The train_flag should be either 'train' or 'test'.")

def read_train_data(file_paths: list, dic: dict):
    """
    Read the training data.
    :param file_paths: a list of file paths.
    :param dic: a dictionary including training apk names.
    :return: a list of training vectors and a list of training labels.
    """
    train_vectors = []
    train_labels = []
    val_vectors = []
    val_labels = []
    test_vectors = []
    test_labels = []

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            csv_data = csv.reader(f)
            for line in islice(csv_data, 1, None):
                sha256 = line[0]
                vector = [float(i) for i in line[1:-1]]
                label = int(float(line[-1]))
                if sha256 in dic['train']:
                    train_vectors.append(vector)
                    train_labels.append(label)
                elif sha256 in dic['valid']:
                    val_vectors.append(vector)
                    val_labels.append(label)
                elif sha256 in dic['test']:
                    test_vectors.append(vector)
                    test_labels.append(label)
                else:
                    continue
    
    return train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels

def read_test_data(file_paths: list, dic: dict):
    """
    Read the test data. (This is design for test data that need to merged together.)
    :param file_paths: a list of file paths.
    :param dic: a dictionary including test apk names.
    :return: a list of test vectors and a list of test labels.
    """
    vectors = []
    labels = []
    test_data = dic['test']

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            csv_data = csv.reader(f)
            for line in islice(csv_data, 1, None):
                sha256 = line[0]
                vector = [float(i) for i in line[1:-1]]
                label = int(float(line[-1]))
                if sha256 in test_data:
                    vectors.append(vector)
                    labels.append(label)
                else:
                    continue
    
    return vectors, labels

def seed_torch(seed=101):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

class Attack_malscan(torch.nn.Module):

	def __init__(self, args):
		super().__init__()
		self.device = torch.device(args.cuda)
		self.data_dir = "../data/Malscan_data/"
		self.feature_file_paths = os.listdir(self.data_dir)
		self.feature_file_paths = [self.data_dir + item for item in self.feature_file_paths]

		self.data_distribution = "../data/ASR/filenames_hash/data_all_attack.json"
		self.logger = logging.getLogger("my_logger")
		self.logger.setLevel(logging.DEBUG)  # Set the logging level to INFO or desired level
		
		base_path = '../weights/attack/Malscan'
		ensureDir(base_path)
		self.handler = logging.FileHandler(os.path.join(base_path, "stats.log"))  # Specify the log file name
		self.formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
		self.handler.setFormatter(self.formatter)
		self.logger.addHandler(self.handler)
		
		self.saveID = args.saveID
		self.model_type = "knn"
		self.model_path = '../weights/Malscan/year=all'
		self.save_path = os.path.join(self.model_path, "malscan.pkl")
		self.white_box = args.white_box

		with open(self.data_distribution, "r") as json_file:
			self.dic = json.load(json_file)
			

	def adversarial_attack(self, **kwargs):
		
		# Load the testing data
		#if type != 'degree':
		#	raise ValueError('Currently, we only support the degree centrality type.')
		train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels = degree_centrality_feature(self.feature_file_paths , train_flag='train', dic=self.dic)
		x_train = np.array(train_vectors, dtype=np.float32)
		x_test = np.array(test_vectors, dtype=np.float32)
		y_test = np.array(test_labels)

		# Load and evaluate the model
		if not os.path.exists(self.save_path):
			raise FileNotFoundError('The model file does not exist.')

		if self.model_type == 'knn':
			model = joblib.load(self.save_path)
			y_pred_train = model.predict(x_train)

			train_data = SimpleDataset(x_train, y_pred_train)
			if self.white_box:
				substitute_model = None
			else:
				substitute_model = MLP(in_channels=x_train.shape[1],
						hidden_channels=1024,
						out_channels=2,
						attention=False)
				# epochs = kwargs['epochs'] if 'epochs' in kwargs else 20
				self.logger.info('Training the substitute model...')
				mlp_train(substitute_model, self.logger, train_data, evaluation=False)
				
			jsma = JSMA(model, self.logger, substitute_model, max_iters=100, attack_model='malscan')
		else:
			raise ValueError(f'The model type {self.model_type} is not supported.')
		
		# evade the model using JSMA
		self.logger.info('Testing the model...')
		self.logger.info('Test samples: {}'.format(x_test.shape[0]))
		y_pred_test = model.predict(x_test)
		#ret_test = eval_metrics(test_labels, y_pred_test)
		#self.logger.info(f'Test: F1: {ret_test["f1"]:.4f}, Precision: {ret_test["precision"]:.4f}, Recall: {ret_test["recall"]:.4f}, Accuracy: {ret_test["accuracy"]:.4f}')
		
		self.logger.info('Attacking the model...')
		x_test = torch.from_numpy(x_test).to(get_device())
		y_test = torch.from_numpy(y_test).to(get_device())
		jsma.attack(x_test, y_test)


if __name__ == '__main__':
    args = parse_args()
    seed_torch(args.seed) 

    Attack = Attack_malscan(args)
    Attack.adversarial_attack(device_id = args.cuda, lr = args.lr, epochs = args.epoch, batch_size = args.batch_size)