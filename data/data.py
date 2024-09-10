"""
This file defines the dataset for MsDroid.
"""
import os
import torch
import json
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class MsdroidDataset():
    def __init__(self, args, root = "graph_data/"):
        
        self.root = root
        self.alpha = args.alpha

        self.train_num_dict = {"2016":16653, "2017":16339, "2018":15696, "2019":15734, "2020":14926, "2016-2019":73523, "all":79348}
        self.train_benign_num_dict = {"2016":14979, "2017":14670, "2018":14056, "2019":14098, "2020":13387, "2016-2019":66004, "all":71190}
        self.train_mal_num_dict = {"2016":1674, "2017":1669, "2018":1640, "2019":1636, "2020":1539, "2016-2019":7519, "all":8158}
        self.valid_num_dict = {"2016":4720, "2017":4732, "2018":4453, "2019":4600, "2020":4437, "2016-2019":18514, "all":22951}
        self.test_num_dict = {"2016":2300, "2017":2322, "2018":2274, "2019":2205, "2020":2093, "2016-2019":21456, "all":11194}

        self.best_valid_f1 = -np.inf
        self.best_valid_epoch, self.patience = 0, 0

        ## get directory name
        self.root_dir = "data/ASR"
        self.file_dir = "filenames"
        self.file_dir_black = "filenames"

        if args.modeltype == 'MsDroid':
            self.file_dir = "filenames_msdroid"

        ## get file name
        self.train_year = args.train_year
        self.concept_drift = args.concept_drift

        # get number of datapoints
        self.n_train = self.train_num_dict[self.train_year]
        self.n_train_benign = self.train_benign_num_dict[self.train_year]
        self.n_train_mal = self.train_mal_num_dict[self.train_year]
        self.n_valid = self.valid_num_dict[self.train_year]
        self.n_test = self.test_num_dict[self.train_year]
        
        if self.concept_drift:
            self.test_year = args.test_year
            self.n_test = self.test_num_dict[self.test_year]
            file_name = "data_" + self.train_year + "_" + self.test_year + ".json"
        else:
            file_name = "data_" + self.train_year + ".json"

        self.json_file_path = os.path.join(self.root_dir, self.file_dir, file_name)
        self.json_file_path_black = os.path.join(self.root_dir, self.file_dir_black, file_name)

        ## read in data distribution
        self.data_distribution = {}

        with open(self.json_file_path, "r") as json_file:
            self.data_distribution = json.load(json_file)

        ## for MsDroid
        if args.modeltype == 'MsDroid' or args.blackboxtype == 'MsDroid':
            with open(self.json_file_path_black, "r") as json_file:
                self.data_distribution_black = json.load(json_file)
 
    def aggregate(data):
  
        tmp = []
        for subgraph in data.data:
            tmp.append(subgraph)
        tmp_apk = DataLoader(tmp, batch_size=len(tmp))
        for batch in tmp_apk:
            data = Data(x = batch.x, edge_index = batch.edge_index, y = batch.y[0], num_nodes = batch.num_nodes)
        
        return data

    def construct_dataset_msdroid(self, indices, mode = "train"):

        dataindex_list = self.data_distribution[mode]

        dataset = []

        data_paths = []
        
        for idx in indices:

            data_path = dataindex_list[idx]

            data_paths.append(data_path)

            data_object = torch.load(data_path)

            for sample in data_object.data:
                dataset.append(sample)
        
        return dataset, data_paths
  
    def construct_dataset_black(self, indices, mode = "train"):

        dataset = []

        dataindex_list = self.data_distribution_black[mode]

        
        for idx in indices:

            data_path = dataindex_list[idx]

            data_object = torch.load(data_path)

            #data_object = self.aggregate(data_object)
            dataset.append(data_object)
        
        return dataset

    def construct_dataset(self, indices, mode = "train"):
        
        dataset = []

        dataname_list = []

        dataindex_list = self.data_distribution[mode]

        for idx in indices:

            if mode == "train_benign":
                idx = int(idx * self.n_train_benign/self.n_train)
            elif mode == "train_malware":
                idx = int(idx * self.n_train_mal/self.n_train)

            data_path = dataindex_list[idx]

            data_object = torch.load(data_path)

            #data_object = self.aggregate(data_object)
            dataset.append(data_object)
            dataname_list.append(data_path)

        return dataset, dataname_list


    def check_size(self, mode = "train"):
        
        dataset = []

        dataindex_list = self.data_distribution[mode]
        
        for filename in dataindex_list:

            #data_path = dataindex_list[idx]

            data_object = torch.load(filename)

            #data_object = self.aggregate(data_object)
            #print(data_object.x.shape[0])
            #if (data_object.x.shape[0]==3558592):

            #    print(dataset.append(data_object.x.shape[0]))
            #    print(filename)

        
        return dataset
          
    def aggregate(self, data):
        """
        Aggregate the subgraphs.
        """
        tmp = []
        for subgraph in data.data:
            tmp.append(subgraph)
        tmp_apk = DataLoader(tmp, batch_size=len(tmp))
        for batch in tmp_apk:
            data = Data(x = batch.x, edge_index = batch.edge_index, y = batch.y[0], num_nodes = batch.num_nodes)
        
        return data
    