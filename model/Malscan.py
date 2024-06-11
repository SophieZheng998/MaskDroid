import csv
import joblib
import numpy as np
import os
import time
import json
from sklearn.neighbors import KNeighborsClassifier
from itertools import islice
from logging import Logger
import torch
from Utils.helper import ensure_dir
from Utils.Network.helper import save_results, random_features, eval_metrics, val_none
from logging import Logger
import logging
import random
import sys
from parse import parse_args
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix


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


class Malscan(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.device = torch.device(args.cuda)
        self.data_dir = "../data/Malscan_data/"
        self.feature_file_paths = os.listdir(self.data_dir)
        self.feature_file_paths = [self.data_dir + item for item in self.feature_file_paths]

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
        self.model_path = '../weights/{}/{}/'.format("Malscan", self.saveID)
        ensureDir(self.model_path)
        self.save_path = os.path.join(self.model_path, "malscan.pkl")

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

        self.n_neighbors = 1

        with open(self.data_distribution, "r") as json_file:
            self.dic = json.load(json_file)

    def read_train_data(self, file_paths: list):
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

        record_test_hashkey = []

        for file_path in file_paths:
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

    def read_test_data(self, file_paths: list):
        """
        Read the test data. (This is design for test data that need to merged together.)
        :param file_paths: a list of file paths.
        :param dic: a dictionary including test apk names.
        :return: a list of test vectors and a list of test labels.
        """
        vectors = []
        labels = []
        test_data = self.dic['test']

        record_test_hashkey = []

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
                        record_test_hashkey.append(sha256)
                    else:
                        continue
        
        return vectors, labels, record_test_hashkey

    def obtain_feature_paths(self, centrality_type: str):
        """
        Obtain the feature paths.
        :param feature_dirs: a list of feature directories.
        :param centrality_type: the type of centrality.
        :return: a list of feature file paths.
        """
        feature_file_paths = []
        for feature_dir in self.feature_file_paths:
            feature_file_path = os.path.join(feature_dir, centrality_type + '_features.csv')
            feature_file_paths.append(feature_file_path)

        return feature_file_paths

    def query_centrality_features(self, feature_file_paths: list):
        """
        Query the centrality feature.
        :param feature_file_paths: a list of feature file paths.
        :param train_flag: the flag of training or not.
        :param dic: a dictionary including apk names.
        """
        if self.train_flag == 'train':
            train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels, record_test_hashkey = self.read_train_data(feature_file_paths)
            return train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels, record_test_hashkey
        elif self.train_flag == 'test':
            vectors, labels, record_test_hashkey = self.read_test_data(feature_file_paths)
            return vectors, labels, record_test_hashkey
        else:
            raise ValueError("The train_flag should be either 'train' or 'test'.")
        
    def degree_centrality_feature(self):
        """
        Query the degree centrality feature.
        :param feature_dirs: a list of feature directories.
        :param train_flag: the flag of training or not.
        :param dic: a dictionary including apk names.
        """
        feature_file_paths = self.obtain_feature_paths('degree')
        return self.query_centrality_features(feature_file_paths)

    def katz_centrality_feature(self):
        """
        Query the Katz centrality feature.
        :param feature_dirs: a list of feature directories.
        :param train_flag: the flag of training or not.
        :param dic: a dictionary including apk names.
        """
        feature_file_paths = self.obtain_feature_paths('katz')
        return self.query_centrality_features(feature_file_paths)

    def closeness_centrality_feature(self):
        """
        Query the closeness centrality feature.
        :param feature_dirs: a list of feature directories.
        :param train_flag: the flag of training or not.
        :param dic: a dictionary including apk names.
        """
        feature_file_paths = self.obtain_feature_paths('closeness')
        return self.query_centrality_features(feature_file_paths)

    def harmonic_centrality_feature(self):
        """
        Query the harmonic centrality feature.
        :param feature_dirs: a list of feature directories.
        :param train_flag: the flag of training or not.
        :param dic: a dictionary including apk names.
        """
        feature_file_paths = self.obtain_feature_paths('harmonic')
        return self.query_centrality_features(feature_file_paths)

    def concatenate_centrality_feature(self):
        """
        Query degree, Katz, closeness, and harmonic centrality features and concatenate them together.
        :param feature_dirs: a list of feature directories.
        :param train_flag: the flag of training or not.
        :param dic: a dictionary including apk names.
        """
        if self.train_flag == 'train':
            degree_train_vectors, degree_train_labels, degree_val_vectors, degree_val_labels, degree_test_vectors, degree_test_labels = self.degree_centrality_feature()
            katz_train_vectors, katz_train_labels, katz_val_vectors, katz_val_labels, katz_test_vectors, katz_test_labels = self.katz_centrality_feature()
            closeness_train_vectors, closeness_train_labels, closeness_val_vectors, closeness_val_labels, closeness_test_vectors, closeness_test_labels = self.closeness_centrality_feature()
            harmonic_train_vectors, harmonic_train_labels, harmonic_val_vectors, harmonic_val_labels, harmonic_test_vectors, harmonic_test_labels = self.harmonic_centrality_feature()

            train_vectors = np.concatenate((degree_train_vectors, katz_train_vectors, closeness_train_vectors, harmonic_train_vectors), axis=1)
            train_labels = degree_train_labels
            val_vectors = np.concatenate((degree_val_vectors, katz_val_vectors, closeness_val_vectors, harmonic_val_vectors), axis=1)
            val_labels = degree_val_labels
            test_vectors = np.concatenate((degree_test_vectors, katz_test_vectors, closeness_test_vectors, harmonic_test_vectors), axis=1)
            test_labels = degree_test_labels

            train_vectors = train_vectors.tolist()

            return train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels
        elif self.train_flag == 'test':
            degree_vectors, degree_labels = self.degree_centrality_feature()
            katz_vectors, katz_labels = self.katz_centrality_feature()
            closeness_vectors, closeness_labels = self.closeness_centrality_feature()
            harmonic_vectors, harmonic_labels = self.harmonic_centrality_feature()

            vectors = np.concatenate((degree_vectors, katz_vectors, closeness_vectors, harmonic_vectors), axis=1)
            labels = degree_labels

            vectors = vectors.tolist()

            return vectors, labels
        else:
            raise ValueError("The train_flag should be either 'train' or 'test'.")

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

    def train(self, train_vectors: list, train_labels: list, val_vectors: list, val_labels: list, test_vectors: list, test_labels: list, record_test_hashkey: list):
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
        :param kwargs: other parameters.
        """
        # Feature randomization
        train_vectors, train_labels = random_features(train_vectors, train_labels)

        # Train/evaluate the model
        self.logger.info('Training the model...')
        ensure_dir(self.model_path)

        if self.model_type == 'knn':
            # train
            n_neighbors = self.n_neighbors
            train_start_time = time.time()
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(train_vectors, train_labels)
            train_end_time = time.time()

            y_pred_val = model.predict(val_vectors)
            ret_val, _ = self.eval_metrics(val_labels, y_pred_val, None)
            # test
            test_start_time = time.time()
            y_pred_test = model.predict(test_vectors)
            ret_test, matching_keys = self.eval_metrics(test_labels, y_pred_test, record_test_hashkey)
            test_end_time = time.time()
            # save
            joblib.dump(model, self.save_path)
        else:
            raise ValueError(f'The model type {self.model_type} is not supported.')
        
        self.logger.debug(f'Training time: {train_end_time - train_start_time:.2f}s.')
        self.logger.debug(f'Testing time: {test_end_time - test_start_time:.2f}s.')
        self.logger.info(f'Saved the model to {self.save_path}.')

        return ret_val, ret_test, matching_keys

    def test(self, test_vectors: list, test_labels: list, record_test_hashkey:list):
        """
        Test the model.
        :param test_vectors: a list of test vectors.
        :param test_labels: a list of test labels.
        :param model_name: the name of the model.
        :param model_type: the type of the model.
        :param logger: the logger.
        :param kwargs: other parameters.
        """
        # Load and evaluate the model
        self.logger.info('Testing the model...')
       
        if self.model_type == 'knn':
            model = joblib.load(self.save_path)
            y_pred_test = model.predict(test_vectors)
            ret_test, matching_keys = self.eval_metrics(test_labels, y_pred_test, record_test_hashkey)
        else:
            raise ValueError(f'The model type {self.model_type} is not supported.')

        return ret_test, matching_keys

    def train_and_test(self,centrality_type: str):

        """
        Train and test the model.
        :param feature_dirs: a list of feature directories.
        :param centrality_type: the type of centrality.
        :param data_distribution: the distribution of data.
        :param train_flag: the flag of training or not.
        :param model_name: the name of the model.
        :param model_type: the type of the model.
        :param result_path: the path of the result.
        :param logger: the logger.
        :param kwargs: other parameters.
        :param result_path: the path of the result.
        """
        # Read data distribution
        if not os.path.exists(self.data_distribution):
            raise ValueError("The data distribution does not exist.")

        # Load features    
        if self.train_flag == 'train':
            if centrality_type == 'degree':
                train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels, record_test_hashkey = self.degree_centrality_feature()
            elif centrality_type == 'katz':
                train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels = self.katz_centrality_feature()
            elif centrality_type == 'harmonic':
                train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels = self.harmonic_centrality_feature()
            elif centrality_type == 'closeness':
                train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels = self.closeness_centrality_feature()
            elif centrality_type == 'concatenate':
                train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels = self.concatenate_centrality_feature()
        else:
            if centrality_type == 'degree':
                test_vectors, test_labels, record_test_hashkey = self.degree_centrality_feature()
            elif centrality_type == 'katz':
                test_vectors, test_labels = self.katz_centrality_feature()
            elif centrality_type == 'harmonic':
                test_vectors, test_labels = self.harmonic_centrality_feature()
            elif centrality_type == 'closeness':
                test_vectors, test_labels = self.closeness_centrality_feature()
            elif centrality_type == 'concatenate':
                test_vectors, test_labels = self.concatenate_centrality_feature()
        
        # Train and test the model
        if self.train_flag == 'train':
            self.logger.info(f'Training samples: {len(train_labels)}, Validation samples: {len(val_labels)}, Test samples: {len(test_labels)}')
            ret_val, ret_test, matching_keys = self.train(train_vectors, train_labels, val_vectors, val_labels, test_vectors, test_labels, record_test_hashkey)
            self.logger.info(f'Validation: precision: {ret_val["precision"]:.4f}, recall: {ret_val["recall"]:.4f}, F1: {ret_val["f1_score"]:.4f}, accuracy: {ret_val["accuracy"]:.4f}')
        else:
            self.logger.info(f'Testing samples: {len(test_labels)}')
            ret_test, matching_keys = self.test(test_vectors, test_labels, record_test_hashkey)

        self.logger.info(f'Test:  precision: {ret_test["precision"]:.4f}, recall: {ret_test["recall"]:.4f}, F1: {ret_test["f1_score"]:.4f}, accuracy: {ret_test["accuracy"]:.4f}')

        if self.need_record:
            with open(os.path.join(self.model_path, "names.json"), 'w') as f:
                json.dump(matching_keys, f)


if __name__ == '__main__':
    args = parse_args()
    seed_torch(args.seed) 

    Malscan = Malscan(args)
    Malscan.train_and_test("degree")