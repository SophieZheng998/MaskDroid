import torch
import os
import numpy as np
import joblib
import csv
import pandas as pd
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

def seed_torch(seed=101):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

class Attack_mamadroid(torch.nn.Module):

	def __init__(self, args):
		super().__init__()
		self.device = torch.device(args.cuda)
		self.data_dir = "../data/Mamadroid_data/"
		self.feature_file_paths = os.listdir(self.data_dir)
		self.feature_file_paths = [self.data_dir + item + "/family.csv" for item in self.feature_file_paths]

		self.data_distribution = "../data/ASR/filenames_hash/data_all_attack.json"
		self.logger = logging.getLogger("my_logger")
		self.logger.setLevel(logging.DEBUG)  # Set the logging level to INFO or desired level
		
		base_path = '../weights/attack/Mamadroid'
		ensureDir(base_path)
		self.handler = logging.FileHandler(os.path.join(base_path, "stats.log"))  # Specify the log file name
		self.formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
		self.handler.setFormatter(self.formatter)
		self.logger.addHandler(self.handler)
		
		self.saveID = args.saveID
		self.model_type = "rf"
		self.model_path = '../weights/Mamadroid/year=all'
		self.save_path = os.path.join(self.model_path, "mamadroid.pkl")
		self.white_box = args.white_box

		with open(self.data_distribution, "r") as json_file:
			self.dic = json.load(json_file)

		self.ordered_vectors = {}


	def read_train_data(self):
		"""
        Read the training data.
        :param file_paths: a list of paths to the training data.
        :param dic: a dictionary including training apk names
        :return: a list of training vectors and a list of training labels.
        """
		train_vectors = []
		train_labels = []
		val_vectors = []
		val_labels = []
		test_vectors = []
		test_labels = []
		
		for file_path in self.feature_file_paths:
			with open(file_path, 'r') as f:
				csv_data = csv.reader(f)
				for line in islice(csv_data, 1, None):
					sha256 = line[0]
					vector = [float(i) for i in line[1:-1]]
					label = int(float(line[-1]))
					if sha256 in self.dic['train']:
						train_vectors.append(vector)
						train_labels.append(label)
					elif sha256 in self.dic['valid']:
						val_vectors.append(vector)
						val_labels.append(label)
					elif sha256 in self.dic['test']:
						test_vectors.append(vector)
						test_labels.append(label)
					else:
						continue
		
		return train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels
	
	def adversarial_attack(self, **kwargs):
		
		# Load the testing data
		train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels = self.read_train_data()
		x_train = np.array(train_vectors, dtype=np.float32)
		x_test = np.array(test_vectors, dtype=np.float32)
		y_test = np.array(test_labels)

		# Load and evaluate the model
		if not os.path.exists(self.save_path):
			raise FileNotFoundError('The model file does not exist.')

		if self.model_type == 'knn' or self.model_type == 'rf':
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
				
			jsma = JSMA(model, self.logger, substitute_model, max_iters=100, attack_model='mamadroid')
		else:
			raise ValueError(f'The model type {self.model_type} is not supported.')
		
		# evade the model using JSMA
		self.logger.info('Testing the model...')
		self.logger.info('Test samples: {}'.format(x_test.shape[0]))
		y_pred_test = model.predict(x_test)
		#ret_test = eval_metrics(test_labels, y_pred_test)
		#self.logger.info(f'Test: F1: {ret_test["f1"]:.4f}, Precision: {ret_test["precision"]:.4f}, Recall: {ret_test["recall"]:.4f}, Accuracy: {ret_test["accuracy"]:.4f}')

		self.logger.info('Attacking the model...')

		#x_test = np.array(x_test, dtype=np.float32)
		#y_test = np.array(y_test)
		x_test = torch.from_numpy(x_test).to(get_device())
		y_test = torch.from_numpy(y_test).to(get_device())
		jsma.attack(x_test, y_test)


if __name__ == '__main__':
	args = parse_args()
	seed_torch(args.seed)
	Attack = Attack_mamadroid(args)
	Attack.adversarial_attack(device_id = args.cuda, lr = args.lr, epochs = args.epoch, batch_size = args.batch_size)