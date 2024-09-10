import joblib
import numpy as np
import os
import torch
import pickle
import time
import warnings
import sys
import random
random.seed(101)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse import parse_args
import json
from sklearn.feature_extraction.text import CountVectorizer as CountV
from sklearn.svm import LinearSVC
from logging import Logger
import logging
from Utils.Network.mlp import MLP, mlp_train, mlp_evaluate
from Utils.helper import ensure_dir
from Utils.Network.helper import save_results, eval_metrics, val_none
from Utils.Network.data import SimpleDataset
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

class Drebin(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.device = torch.device(args.cuda)
        self.data_dir = "../data/Drebin/"
        self.feature_dirs = os.listdir(self.data_dir)
        self.feature_dirs = [self.data_dir + item for item in self.feature_dirs]

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
        self.model_path = '../weights/{}/{}/'.format("Drebin", self.saveID)
        ensureDir(self.model_path)
        self.save_path = os.path.join(self.model_path, "drebin.pkl")

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

        # svm or mlp
        self.max_iter = 300
        self.vectorizer_path = os.path.join(self.model_path, "vectorizers.pkl")

        with open(self.data_distribution, "r") as json_file:
            self.dic = json.load(json_file)

        indices = list(range(len(self.dic["train"])))
        sampled_indices = random.sample(indices, len(indices) // 8)
        self.dic["train"] = [self.dic["train"][i] for i in sampled_indices]

        indices = list(range(len(self.dic["valid"])))
        sampled_indices = random.sample(indices, len(indices) // 8)
        self.dic["valid"] = [self.dic["valid"][i] for i in sampled_indices]


    def list_train_files(self):
        """
        List the data files.
        :param feature_dirs: a list of paths to the feature directories.
        :param dic: a dictionary including apk names
        :return: a list of paths to the train data files.
        """
        train_files = []
        val_files = []
        test_files = []

        record_test_hashkey = []

        for feature_dir in self.feature_dirs:
            for file in os.listdir(feature_dir):
                if file.endswith('.data'):
                    file_path = os.path.join(feature_dir, file)
                    if file[:-5] in self.dic['train']:
                        train_files.append(file_path)
                    elif file[:-5] in self.dic['valid']:
                        val_files.append(file_path)
                    elif file[:-5] in self.dic['test']:
                        test_files.append(file_path)
                        record_test_hashkey.append(file[:-5])
                    else:
                        continue
        
        return train_files, val_files, test_files, record_test_hashkey


    def list_test_files(self):
        """
        List the data files.
        :param feature_dirs: a list of paths to the feature directories.
        :param dic: a dictionary including apk names
        :return: a list of paths to the test data files.
        """
        test_files = []

        record_test_hashkey = []

        for feature_dir in self.feature_dirs:
            for file in os.listdir(feature_dir):
                if file.endswith('.data'):
                    file_path = os.path.join(feature_dir, file)
                    if file[:-5] in self.dic['test']:
                        test_files.append(file_path)
                        record_test_hashkey.append(file[:-5])
                    else:
                        continue
        
        return test_files, record_test_hashkey


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


    def train(self, train_files: list, val_files: list, test_files: list, record_test_hashkey:list):
        """
        Train the model.
        :param train_files: a list of paths to the training data files.
        :param val_files: a list of paths to the validation data files.
        :param test_files: a list of paths to the test data files.
        :param model_name: the name of the model.
        :param model_type: the type of the model.
        :param logger: the logger.
        :param kwargs: other arguments.
        :return: validation and test results.
        """
        # Feature encoding
        self.logger.info('Feature engineering...')
        vectorizer = CountV(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None, stop_words=['label=0', 'label=1'], binary=True)
        vectorizer.fit(train_files)
        
        self.logger.info("Storing feature vectors to %s" % self.vectorizer_path)
        ensure_dir(self.vectorizer_path)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer.vocabulary_, f)

        # Construct the training/validation/testing sets
        x_train = vectorizer.transform(train_files).toarray().astype(np.float32)
        x_val = vectorizer.transform(val_files).toarray().astype(np.float32)
        x_test = vectorizer.transform(test_files).toarray().astype(np.float32)
        
        y_train = []
        for file in train_files:
            with open(file, 'r') as f:
                fisrtline = f.readline().strip()
                y_train.append(int(fisrtline.split('=')[-1]))
        y_train = np.array(y_train)

        y_val = []
        for file in val_files:
            with open(file, 'r') as f:
                fisrtline = f.readline().strip()
                y_val.append(int(fisrtline.split('=')[-1]))
        y_val = np.array(y_val)

        y_test = []
        for file in test_files:
            with open(file, 'r') as f:
                fisrtline = f.readline().strip()
                y_test.append(int(fisrtline.split('=')[-1]))
        y_test = np.array(y_test)

        # Train/evaluate the model
        self.logger.info('Training the model...')
        ensure_dir(self.model_path)

        if self.model_type == 'svm':
            # Train
            max_iter = self.max_iter
            train_start_time = time.time()
            model = LinearSVC(max_iter=max_iter)
            model.fit(x_train, y_train)
            train_end_time = time.time()

            # Validation
            if len(y_val) == 0:
                ret_val = val_none()
            else:
                y_pred_val = model.predict(x_val)
                ret_val, _ = self.eval_metrics(y_val, y_pred_val, None)

            # Test
            test_start_time = time.time()
            y_pred_test = model.predict(x_test)
            ret_test, matching_keys = self.eval_metrics(y_test, y_pred_test, record_test_hashkey)
            test_end_time = time.time()

            # Save
            joblib.dump(model, self.save_path)
            self.logger.debug(f'Training time: {train_end_time - train_start_time:.2f}s')
            self.logger.debug(f'Testing time: {test_end_time - test_start_time:.2f}s')

        elif self.model_type == 'mlp':
            # Train & evaluation & save
            model = MLP(in_channels=x_train.shape[1],
                        hidden_channels=1024,
                        out_channels=2,
                        attention=False)
            train_data = SimpleDataset(x_train, y_train)
            val_data = SimpleDataset(x_val, y_val)
            test_data = SimpleDataset(x_test, y_test)
            epochs = self.epochs
            ret_val, ret_test = mlp_train(model, self.logger, train_data, val_data, test_data, self.save_path)
        else:
            raise ValueError('Model type is not valid.')

        self.logger.info(f'Saved the model to {self.save_path}')

        return ret_val, ret_test, matching_keys

    def test(self, test_files: list, record_test_hashkey:list):
        """
        Test the model.
        :param test_files: a list of paths to the test data files.
        :param model_name: the name of the model.
        :param logger: the logger.
        :return: the F1 score, precision, recall, accuracy, and AUC of the model.
        """
        # Feature encoding
        #vectorizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vectorizers', model_name)
        self.logger.info("Loading feature vectors from %s" % self.vectorizer_path)
        vectorizer = CountV(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None, stop_words=['label=0', 'label=1'], binary=True,
                            vocabulary=pickle.load(open(self.vectorizer_path, 'rb')))

        # Construct the testing set
        x_test = vectorizer.transform(test_files).toarray().astype(np.float32)
        y_test = []
        for file in test_files:
            with open(file, 'r') as f:
                fisrtline = f.readline().strip()
                y_test.append(int(fisrtline.split('=')[-1]))
        y_test = np.array(y_test)
        
        # Load and evaluate models
        self.logger.info('Testing the model...')
        #model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', model_name)
        if self.model_type == 'svm':
            model = joblib.load(self.save_path)
            y_pred_test = model.predict(x_test)
            ret_test, matching_keys = self.eval_metrics(y_test, y_pred_test, record_test_hashkey)
        elif self.model_type == 'mlp':
            model = torch.load(self.save_path)
            test_data = SimpleDataset(x_test, y_test)
            ret_test = mlp_evaluate(test_data, model)
        else:
            raise ValueError('Model type is not valid.')

        return ret_test, matching_keys

    def train_and_test(self):
        """
        Train and test the model (SVM).
        :param feature_dirs: a list of paths to the feature directories.
        :param data_distribution: the data distribution.
        :param train_flag: the flag to indicate whether to train the model.
        :param model_name: the name of the model.
        :param model_type: the type of the model.
        :param result_path: the path to the result directory.
        :param logger: the logger.
        :param kwargs: the keyword arguments.
        """
        # Read data distribution
        #if not os.path.exists(data_distribution):
        #    raise FileNotFoundError('The data distribution file does not exist.')
        #with open(data_distribution, 'r') as f:
        #    dic = eval(f.read())
        
        # Get files
        if self.train_flag == 'train':
            train_files, val_files, test_files, record_test_hashkey = self.list_train_files()
        elif self.train_flag == 'test':
            test_files, record_test_hashkey = self.list_test_files()
        else:
            raise ValueError('The train flag is not valid.')
        
        # Train and test the model
        if self.train_flag == 'train':
            self.logger.info("Training samples: {}, validation samples: {}, test samples: {}".format(len(train_files), len(val_files), len(test_files)))
            ret_val, ret_test, matching_keys = self.train(train_files, val_files, test_files, record_test_hashkey)
            self.logger.info(f'Validation: precision: {ret_val["precision"]:.4f}, recall: {ret_val["recall"]:.4f}, F1: {ret_val["f1_score"]:.4f}, accuracy: {ret_val["accuracy"]:.4f}')
        else:
            self.logger.info("Test samples: {}".format(len(test_files)))
            ret_test, matching_keys = self.test(test_files, record_test_hashkey)
        
        self.logger.info(f'Test:  precision: {ret_test["precision"]:.4f}, recall: {ret_test["recall"]:.4f}, F1: {ret_test["f1_score"]:.4f}, accuracy: {ret_test["accuracy"]:.4f}')
 
        if self.need_record:
            with open(os.path.join(self.model_path, "names.json"), 'w') as f:
                json.dump(matching_keys, f)


if __name__ == '__main__':
    args = parse_args()
    seed_torch(args.seed) 

    Drebin = Drebin(args)
    Drebin.train_and_test()