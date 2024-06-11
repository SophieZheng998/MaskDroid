import csv
import joblib
import os
import time
import torch
import sys
import random
import numpy as np
import json
from parse import parse_args
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from itertools import islice
from logging import Logger
import logging
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

#from Utils.helper import ensure_dir
#from Utils.Network.helper import save_results, random_features, eval_metrics, val_none


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


class Mamadroid(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = torch.device(args.cuda)
        self.data_dir = "../data/Mamadroid_data/"
        self.feature_file_paths = os.listdir(self.data_dir)
        self.feature_file_paths = [self.data_dir + item + "/family.csv" for item in self.feature_file_paths]
        
        ## get directory name
        self.root_dir = "../data/ASR/filenames_hash"
        self.train_year = args.train_year
        self.concept_drift = args.concept_drift
        self.need_record = args.need_record

        if self.concept_drift:
            self.test_year = args.test_year
            distribution_file_name = "data_" + self.train_year + "_" + self.test_year + ".json"
        else:
            self.test_year = self.train_year
            distribution_file_name = "data_" + self.train_year + ".json"

        self.data_distribution = os.path.join(self.root_dir, distribution_file_name)
        self.train_flag = args.train_flag

        self.saveID = args.saveID
        self.saveID +=  "year=" + str(args.train_year) 
        self.model_type = args.modeltype
        self.model_path = '../weights/{}/{}/'.format("Mamadroid", self.saveID)
        ensureDir(self.model_path)
        self.save_path = os.path.join(self.model_path, "mamadroid.pkl")

        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(logging.DEBUG)  # Set the logging level to INFO or desired level

        # Create a handler and formatter
        if self.concept_drift:
            self.handler = logging.FileHandler(os.path.join(self.model_path, "{}_stats.log".format(self.test_year)))
        else:
            self.handler = logging.FileHandler(os.path.join(self.model_path, "stats.log"))  # Specify the log file name
        self.formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

        self.n_neighbors = 5

        with open(self.data_distribution, "r") as json_file:
            self.dic = json.load(json_file)
        
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

        record_test_hashkey = []

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
                        record_test_hashkey.append(sha256)
                    else:
                        continue
        
        return train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels, record_test_hashkey

    def read_test_data(self):
        """
        Read the test data.
        :param file_paths: a list of paths to the test data.
        :param dic: a dictionary including test apk names
        :return: a list of test vectors and a list of test labels.
        """
        test_vectors = []
        test_labels = []

        record_test_hashkey = []

        test_data = self.dic['test']
        for file_path in self.feature_file_paths:
            with open(file_path, 'r') as f:
                csv_data = csv.reader(f)
                for line in islice(csv_data, 1, None):
                    sha256 = line[0]
                    vector = [float(i) for i in line[1:-1]]
                    label = int(float(line[-1]))
                    if sha256 in test_data:
                        test_vectors.append(vector)
                        test_labels.append(label)
                        record_test_hashkey.append(sha256)
                    else:
                        continue
        
        return test_vectors, test_labels, record_test_hashkey

    def eval_metrics(self, true_labels, predicted_labels, record_test_hashkey):

        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        accuracy = accuracy_score(true_labels, predicted_labels)

        n_ret = {"precision": precision, "recall": recall, "f1_score": f1, "accuracy": accuracy}

        if record_test_hashkey is not None:
            matching_keys = [record_test_hashkey[i] for i in range(len(true_labels)) 
                        if true_labels[i] == 1 and predicted_labels[i] == 1]
        else:
            matching_keys = None

        return n_ret, matching_keys

    def train(self, train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels, record_test_hashkey):
        """
        Train the model.
        :param train_vectors: a list of training vectors.
        :param train_labels: a list of training labels.
        :param val_vectors: a list of validation vectors.
        :param val_labels: a list of validation labels.
        :param test_vectors: a list of test vectors.
        :param test_labels: a list of test labels.
        :param model_name: the name of the model.
        :param model_type: the type of the model.
        :param logger: the logger.
        """

        # Feature randomization
        # TODO
        #train_vectors, train_labels = random_features(train_vectors, train_labels)

        # Train/evaluate the model
        self.logger.info('Training the model...')
        
        ensureDir(self.model_path)

        if self.model_type == 'knn':
            # Train
            n_neighbors = self.n_neighbors
            train_start_time = time.time()
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(train_vectors, train_labels)
            train_end_time = time.time()
            
            y_pred_val = model.predict(val_vectors)
            ret_val, _ = self.eval_metrics(val_labels, y_pred_val)
            # Test
            test_start_time = time.time()
            y_pred_test = model.predict(test_vectors)
            ret_test, matching_keys = self.eval_metrics(test_labels, y_pred_test)

            test_end_time = time.time()  
            # Save
            joblib.dump(model, self.save_path)
        elif self.model_type == 'rf':
            # Train
            train_start_time = time.time()
            model = RandomForestClassifier()
            model.fit(train_vectors, train_labels)
            train_end_time = time.time()
            
            y_pred_val = model.predict(val_vectors)
            ret_val, _ = self.eval_metrics(val_labels, y_pred_val, None)
            # Test
            test_start_time = time.time()
            y_pred_test = model.predict(test_vectors)
            ret_test, matching_keys = self.eval_metrics(test_labels, y_pred_test, record_test_hashkey)
            test_end_time = time.time()
            # Save
            joblib.dump(model, self.save_path)
        else:
            raise ValueError(f'The model type {self.model_type} is not supported.')
        
        self.logger.debug(f'Training time: {train_end_time - train_start_time:.2f}s.')
        self.logger.debug(f'Testing time: {test_end_time - test_start_time:.2f}s.')
        self.logger.info(f'Saved the model to {self.model_path}.')

        return ret_val, ret_test,  matching_keys

    def test(self, test_vectors, test_labels, record_test_hashkey):
        """
        Test the model.
        :param test_vectors: a list of test vectors.
        :param test_labels: a list of test labels.
        :param model_name: the name of the model.
        :param model_type: the type of the model.
        :param logger: the logger.
        """
        # Load and evaluate the model
        self.logger.info('Testing the model...')
        
        if self.model_type == 'knn' or self.model_type == 'rf':
            model = joblib.load(self.save_path)
            y_pred_test = model.predict(test_vectors)
            ret_test, matching_keys = self.eval_metrics(test_labels, y_pred_test, record_test_hashkey)
        else:
            raise ValueError(f'The model type {self.model_type} is not supported.')
        
        return ret_test, matching_keys

    def train_and_test(self):
        """
        Train and test the model.
        :param feature_dirs: a list of paths to the feature directories.
        :param data_distribution: the data distribution.
        :param train_flag: the flag to train the model.
        :param model_name: the name of the model.
        :param model_type: the type of the model.
        :param result_path: the path to the result file.
        :param logger: the logger.
        """
        # Read data distribution
        if not os.path.exists(self.data_distribution):
            raise FileNotFoundError('The data distribution file does not exist.')
        with open(self.data_distribution, 'r') as f:
            self.dic = eval(f.read())

        # Load the training data
        if self.train_flag == 'train':
            train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels, record_test_hashkey = self.read_train_data()
        elif self.train_flag == 'test':
            test_vectors, test_labels, record_test_hashkey = self.read_test_data()
        else:
            raise ValueError('The train flag is invalid.')

        # Train and test the model
        if self.train_flag == 'train':
            self.logger.info(f'Training samples: {len(train_vectors)}, validation samples: {len(val_vectors)}, test samples: {len(test_vectors)}')

            ret_val, ret_test, matching_keys = self.train(train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels, record_test_hashkey)
            self.logger.info(f'Validation: precision: {ret_val["precision"]:.4f}, recall: {ret_val["recall"]:.4f}, F1: {ret_val["f1_score"]:.4f}, accuracy: {ret_val["accuracy"]:.4f}')
        else:
            self.logger.info(f'Testing samples: {len(test_vectors)}')
            ret_test, matching_keys = self.test(test_vectors, test_labels, record_test_hashkey)

        self.logger.info(f'Test:  precision: {ret_test["precision"]:.4f}, recall: {ret_test["recall"]:.4f}, F1: {ret_test["f1_score"]:.4f}, accuracy: {ret_test["accuracy"]:.4f}')

        if self.need_record:
            with open(os.path.join(self.model_path, "names.json"), 'w') as f:
                json.dump(matching_keys, f)


if __name__ == '__main__':
    args = parse_args()
    seed_torch(args.seed) 

    Mamadroid = Mamadroid(args)
    Mamadroid.train_and_test()

