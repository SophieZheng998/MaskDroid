
import os
import re
import time
import datetime
import pandas as pd
import torch
import json
import torch.nn as nn
from torch_geometric.data import Batch
from data.data import MsdroidDataset
from torch_geometric.loader import DataLoader
from model.base.utils import *
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix


# define the abstract class for recommender system
class AbstractRUN(nn.Module):
    def __init__(self, args) -> None:
        super(AbstractRUN, self).__init__()

        self.args = args
        self.cuda = args.cuda
        self.device = torch.device(args.cuda)
        self.patience = args.patience
        self.lr = args.lr
        self.reg = args.reg
        self.batch_size = args.batch_size
        self.max_epoch = args.epoch
        self.verbose = args.verbose
        self.modeltype = args.modeltype
        self.criterion = torch.nn.NLLLoss()
        self.need_record = args.need_record
        self.sh = args.sh
        self.restore_epoch = args.restore_epoch

        # load the data
        self.data = MsdroidDataset(args) # load data from the path

        # load the 
        #self.running_model = args.modeltype + '_batch' if self.inbatch else args.modeltype
        exec('from model.'+ args.modeltype + ' import ' + args.modeltype) # import the model first
        self.model = eval(args.modeltype + '(args)') # initialize the model with the graph
        self.model.cuda(self.device)

        # preparing for saving
        self.preperation_for_saving(args)

    # the whole pipeline of the training process
    def execute(self):

        # write args
        perf_str = str(self.args)
        with open(self.base_path + 'stats.txt','a') as f:
            f.write(perf_str+"\n")

        self.model, self.start_epoch = self.restore_checkpoint(self.model, self.base_path, self.cuda) 

        if self.args.need_pretrain:
            print("start training") 
            self.train()
            self.model = self.restore_best_checkpoint(self.data.best_valid_epoch, self.model, self.base_path, self.cuda)
        
        print("start testing")
        self.model.eval() # evaluate the best model
        print_str = "The best epoch is % d" % self.data.best_valid_epoch
        with open(self.base_path +'stats.txt', 'a') as f:
            f.write(print_str + "\n")

        _, _, final_train = self.evaluation(self.args, self.data, self.model, self.data.best_valid_epoch, self.base_path, name = "train")
        _, _, final_valid = self.evaluation(self.args, self.data, self.model, self.data.best_valid_epoch, self.base_path, name = "valid")
        _, _, final_test = self.evaluation(self.args, self.data, self.model, self.data.best_valid_epoch, self.base_path, name = "test")

    # define the training process
    def train(self) -> None:
        
        self.set_optimizer() # get the optimizer
        self.flag = False
        for epoch in range(self.start_epoch, self.max_epoch):
            # print(self.model.embed_user.weight)
            if self.flag: # early stop
                break
            # All models
            t1 = time.time()
            losses = self.train_one_epoch(epoch) # train one epoch
            t2 = time.time()
            self.document_running_loss(losses, epoch, t2-t1) # report the loss
            if (epoch + 1) % self.verbose == 0:
            #if (epoch + 1) % 30 == 0: # evaluate the model
                self.eval_and_check_early_stop(epoch)     

    #! must be implemented by the subclass
    def train_one_epoch(self, epoch):
        raise NotImplementedError
 
    def preperation_for_saving(self, args):

        #self.formatted_today=datetime.date.today().strftime('%m%d') + '_'

        self.saveID =  args.saveID  

        #if args.contrastive:
        #    self.saveID += "_tau=" + str(args.tau)

        self.saveID +=  "year=" + str(args.train_year) + "_lr=" + str(args.lr) + "_h=" + str(args.hidden_channels) + \
                        "_drop=" + str(args.dropout_ratio) + "_ms=" + str(args.mask_rate)    
        
        self.modify_saveID()

        self.base_path = './weights/{}/{}'.format(self.modeltype, self.saveID)
        
        self.checkpoint_buffer=[]

        ensureDir(self.base_path)
 
    def modify_saveID(self):
        pass

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam([param for param in self.model.parameters() if param.requires_grad == True], betas=(0.9, 0.98), lr=self.lr, eps=1e-6, weight_decay = self.reg)

     # load the checkpoint

    def restore_checkpoint(self, model, checkpoint_dir, device):
        """
        If a checkpoint exists, restores the PyTorch model from the checkpoint.
        Returns the model and the current epoch.
        """
        cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                    if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

        if not cp_files:
            print('No saved model parameters found')
            return model, 0,

        epoch_list = []

        regex = re.compile(r'\d+')

        for cp in cp_files:
            epoch_list.append([int(x) for x in regex.findall(cp)][0])

        epoch = max(epoch_list)

        if self.sh:
            inp_epoch = self.restore_epoch
        else:
            print("Which epoch to load from? Choose in range [0, {})."
                    .format(epoch), "Enter 0 to train from scratch.")
            print(">> ", end = '')
            inp_epoch = int(input())

        if self.args.clear_checkpoints:
            print("Clear checkpoint")
            clear_checkpoint(checkpoint_dir)
            return model, 0,

        #inp_epoch = epoch
        if inp_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0,
        

        filename = os.path.join(checkpoint_dir,
                                'epoch={}.checkpoint.pth.tar'.format(inp_epoch))

        print("Loading from checkpoint {}?".format(filename))

        try:
            checkpoint = torch.load(filename, str(device))
        except:
            checkpoint = torch.load(filename, map_location="cuda:{}".format(device))

        try:
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Successfully restored checkpoint (trained for {} epochs)"
                .format(checkpoint['epoch']))
        except:
            print("=> Checkpoint not successfully restored")
            raise

        return model, inp_epoch
    
    def document_running_loss(self, losses:list, epoch, t_one_epoch):
        loss_str = ', '.join(['%.5f']*len(losses)) % tuple(losses)
        perf_str = 'Epoch %d [%.1fs]: train==[' % (
                epoch, t_one_epoch) + loss_str + ']'
        with open(self.base_path + 'stats.txt','a') as f:
                f.write(perf_str+"\n")
    
    def eval_and_check_early_stop(self, epoch):
        self.model.eval()

        is_best, temp_flag, _  = self.evaluation(self.args, self.data, self.model, epoch, self.base_path)
            
        if is_best:
            
            save_checkpoint(self.model, epoch, self.base_path, self.checkpoint_buffer, self.args.max2keep)
            

        if temp_flag:
            self.flag = True

        self.model.train()     

    def evaluation(self, args, data, model, epoch, base_path, name = "valid"):

        if name == "train":
            evaluate_idx = list(range(self.data.n_train))
        elif name == "valid":
            evaluate_idx = list(range(self.data.n_valid))
        elif name == "test":
            evaluate_idx = list(range(self.data.n_test))

        eval_loader = DataLoader(evaluate_idx, batch_size=self.batch_size, shuffle=False)

        true_labels = []
        predicted_labels = []
        names = []

        with torch.no_grad():
            for batch in eval_loader:
                #batch = [x.cuda(self.device) for x in batch]

                evaluate_objects, datapath_list = self.data.construct_dataset(batch, name)
                batch = Batch.from_data_list(evaluate_objects)

                batch.cuda(self.device)
                output, _  = model(batch.x, batch.edge_index, batch.batch, batch.y)
                predictions = output.argmax(dim=1)
                true_labels.extend(batch.y.cpu().tolist())
                predicted_labels.extend(predictions.cpu().tolist())

                if name == "test" and self.need_record:
                    for true_label, predicted_label, datapath in zip(batch.y.cpu().tolist(), predictions.cpu().tolist(), datapath_list):
                        if true_label == 1 and predicted_label == 1:
                            names.append(datapath)

        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        accuracy = accuracy_score(true_labels, predicted_labels)

        n_ret = {"precision": precision, "recall": recall, "f1_score": f1, "accuracy": accuracy}

        perf_str = name+':{}'.format(n_ret)
        print(perf_str)
        with open(base_path + 'stats.txt', 'a') as f:
            f.write(perf_str + "\n")

        if name == "test" and self.need_record:
            with open(base_path + '_names.json', 'w') as f:
                json.dump(names, f)

        # Check if need to early stop (on validation)
        is_best=False
        early_stop=False

        if name=="valid":
            if f1 > data.best_valid_f1:
                data.best_valid_epoch = epoch
                data.best_valid_f1 = f1
                data.patience = 0
                is_best=True
            else:
                data.patience += 1
                if data.patience >= args.patience:
                    print_str = "The best performance epoch is % d " % data.best_valid_epoch
                    print(print_str)
                    early_stop=True

        return is_best, early_stop, n_ret

    def restore_best_checkpoint(self, epoch, model, checkpoint_dir, device):
        """
        Restore the best performance checkpoint
        """
        filename = os.path.join(checkpoint_dir,
                                'epoch={}.checkpoint.pth.tar'.format(epoch))

        print("Loading from checkpoint {}?".format(filename))

        try:
            checkpoint = torch.load(filename, str(device))
        except:
            checkpoint = torch.load(filename, map_location="cuda:{}".format(device))

        model.load_state_dict(checkpoint['state_dict'])
        print("=> Successfully restored checkpoint (trained for {} epochs)"
            .format(checkpoint['epoch']))

        return model
    
    def document_hyper_params_results(self, base_path, final_valid, final_test):

        overall_path = '/'.join(base_path.split('/')[:-1]) + '/'
        hyper_params_results_path = overall_path + '_' + self.modeltype + '_' + self.args.saveID + '_hyper_params_results.csv'

        results = {'notation': self.formatted_today, 'best_epoch': max(self.data.best_valid_epoch, self.start_epoch), \
                   'batch_size': self.batch_size, 'lr': self.lr,
                   'hidden_channels': self.args.hidden_channels,
                   'dropout_ratio': self.args.dropout_ratio,
                   'gnn_type': self.args.gnn_type, "JK" : self.args.JK}
        
        for k, v in final_valid.items():
            results["valid_" + k] = round(v, 4)
        
        for k, v in final_test.items():
            results["test_" + k] = round(v, 4)
        
        frame_columns = list(results.keys())
        
        if os.path.exists(hyper_params_results_path):
            hyper_params_results = pd.read_csv(hyper_params_results_path)
        else:
            hyper_params_results = pd.DataFrame(columns=frame_columns)

    
        results = pd.DataFrame([results])
        hyper_params_results = pd.concat([hyper_params_results, results])
        
        hyper_params_results.to_csv(hyper_params_results_path, index=False, float_format='%.4f')
        
