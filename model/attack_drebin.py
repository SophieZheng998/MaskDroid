import torch
import os
import pickle
import numpy as np
import joblib
import sys
import random
import logging
import json
from sklearn.feature_extraction.text import CountVectorizer as CountV
from logging import Logger
from Utils.Network.mlp import MLP, mlp_train
from Utils.Network.jsma_cml import JSMA
from Utils.Network.data import SimpleDataset
from Utils.Network.helper import get_device
from parse import parse_args

def seed_torch(seed=101):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def ensureDir(dir_path):
     if not os.path.exists(dir_path):
          os.makedirs(dir_path)

class Attack_drebin(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.device = torch.device(args.cuda)
        self.data_dir = "../data/Drebin/"
        self.feature_dirs = os.listdir(self.data_dir)
        self.feature_dirs = [self.data_dir + item for item in self.feature_dirs]
        self.data_distribution = "../data/ASR/filenames_hash/data_all_attack.json"

        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(logging.DEBUG)  # Set the logging level to INFO or desired level

        # Create a handler and formatter
        base_path = '../weights/attack/Drebin'
        ensureDir(base_path)
        self.handler = logging.FileHandler(os.path.join(base_path, "stats.log"))  # Specify the log file name
        self.formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        self.cuda = args.cuda
        self.saveID = args.saveID
        # svm or mlp
        self.max_iter = 300
        self.epochs = 20
        self.model_type = "svm"
        self.save_path = '../weights/Drebin/year=all'
        self.model_path = os.path.join(self.save_path, "drebin.pkl")
        self.vectorizer_path = os.path.join(self.save_path, "vectorizers.pkl")

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
                    else:
                        continue
        
        return train_files, val_files, test_files, 


    def adversarial_attack(self):
        
        # Get traing/test files
        train_files, _, test_files = self.list_train_files()
        self.logger.info("Train samples: {} (used in black-box attacks)".format(len(train_files)))
        self.logger.info("Test samples: {}".format(len(test_files)))

        # Feature encoding
        self.logger.info("Loading feature vectors from %s" % self.vectorizer_path)
        vectorizer = CountV(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None, stop_words=['label=0', 'label=1'], binary=True)
        vectorizer.fit(train_files)
        
        # construct the training set (no need to load y_train in black-box attacks)
        x_train = vectorizer.transform(train_files).toarray().astype(np.float32)

        # construct the testing set
        x_test = vectorizer.transform(test_files).toarray().astype(np.float32)
        y_test = []
        for file in test_files:
            with open(file, 'r') as f:
                fisrtline = f.readline().strip()
                y_test.append(int(fisrtline.split('=')[-1]))
        y_test = np.array(y_test)

        # load models
        
        self.logger.info('Loading the model from %s' % self.model_path)
        if self.model_type == 'svm':
            model = joblib.load(self.model_path)
            y_pred_train = model.predict(x_train)
            train_data = SimpleDataset(x_train, y_pred_train)
            substitute_model = MLP(in_channels=x_train.shape[1],
                        hidden_channels=1024,
                        out_channels=2,
                        attention=False)
            # epochs = kwargs['epochs'] if 'epochs' in kwargs else 20
            self.logger.info('Training the substitute model...')
            mlp_train(substitute_model, self.logger, train_data, evaluation=False,  device = self.cuda)
            jsma = JSMA(model, self.logger, substitute_model, attack_model='drebin', device = self.cuda)
        elif self.model_type == 'mlp':
            model = torch.load(self.model_path)
            jsma = JSMA(model, self.logger, substitute_model=None, attack_model='drebin', device = self.cuda)
        else:
            raise ValueError('Model type is not valid.')
        
        # evade the model using JSMA
        self.logger.info('Attacking the model...')
        x_test = torch.from_numpy(x_test).to(get_device(self.cuda))
        y_test = torch.from_numpy(y_test).to(get_device(self.cuda))
        jsma.attack(x_test, y_test)


if __name__ == '__main__':
	args = parse_args()
	seed_torch(args.seed)
	Attack = Attack_drebin(args)
	Attack.adversarial_attack()