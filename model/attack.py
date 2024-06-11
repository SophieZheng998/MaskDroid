import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import logging
import sys
from model.GNN import GNN
from model.PreModel import PreModel
from model.PreModel_v3 import PreModel_v3
from model.PreModel_v4 import PreModel_v4
from parse import parse_args
from model.jsma_new import JSMA_new
#from model.jsma_new import JSMA_vae
from model.jsma_msdroid import JSMA_msdroid
from model.jsma_gnn_v2 import JSMA_gnn_v2
from model.jsma_gnn import JSMA
from model.jsma_c2 import JSMA_c2
from model.GNN_v2 import GNN_v2
from model.GNN_VAE import GNN_VAE
from model.BlackBox import BlackBox
from model.MsDroid import MsDroid
from model.jsma_gnn_vae import JSMA_gnn_vae

import random
import os
import numpy as np

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


if __name__ == '__main__':

     # load in pretrained model
     args = parse_args()
     seed_torch(args.seed) 
     device = torch.device(args.cuda)

    # Create a Logger instance
     logger = logging.getLogger("my_logger")
     logger.setLevel(logging.DEBUG)  # Set the logging level to INFO or desired level

    # Create a handler and formatter
     base_path = '../weights/attack/{}/{}'.format(args.modeltype, args.saveID)
     ensureDir(base_path)
     if args.white_box:
          handler = logging.FileHandler(os.path.join(base_path, "stats.log"))
     else:
          handler = logging.FileHandler(os.path.join(base_path, "black_stats_new.log"))

     formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
     handler.setFormatter(formatter)
     logger.addHandler(handler)

     if args.modeltype == 'PreModel_v3':
          model = PreModel_v3(args)
          if args.saveID == 'rate=0_2':
               checkpoint_path = "../weights/PreModel_v3/rate=0_2year=2020_lr=0.001_h=64_drop=0.2_ms=0.2/epoch=19.checkpoint.pth.tar"
          elif args.saveID == 'rate=0_5':
               checkpoint_path = "../weights/PreModel_v3/rate=0_5year=2020_lr=0.001_h=64_drop=0.2_ms=0.5/epoch=19.checkpoint.pth.tar"
          elif args.saveID == 'rate=0_8':
               checkpoint_path = "..weights/PreModel_v3/year=2020_lr=0.001_h=64_drop=0.2_ms=0.8/epoch=17.checkpoint.pth.tar"
          elif args.saveID == 'rate=0_9':
               checkpoint_path = "../weights/PreModel_v3/rate=0_9year=2020_lr=0.001_h=64_drop=0.2_ms=0.9/epoch=11.checkpoint.pth.tar"
          else:
               checkpoint_path = "../weights/PreModel_v3/year=all_lr=0.001_h=64_drop=0.2_ms=0.8/epoch=10.checkpoint.pth.tar"
               checkpoint_path_sub = "../weights/BlackBox/MaskDroidyear=all_lr=0.001_h=64_drop=0.2_ms=0.8/epoch=13.checkpoint.pth.tar"
     elif args.modeltype == 'GNN_VAE':
          model = GNN_VAE(args)
          checkpoint_path = "../weights/GNN_VAE/year=all_lr=0.001_h=64_drop=0.2_ms=0.8/epoch=8.checkpoint.pth.tar"    
          checkpoint_path_sub = "../weights/BlackBox/FDVAEyear=all_lr=0.001_h=64_drop=0.2_ms=0.8/epoch=8.checkpoint.pth.tar"
     elif args.modeltype == 'MsDroid':
          model = MsDroid(args)
          checkpoint_path = "..old/weights_old/MsDroid/REBUTTAL_lr=0.001_h=64_drop=0.2_ms=0.8/epoch=7.checkpoint.pth.tar"
          checkpoint_path_sub = "..old/weights_old/BlackBox/REBUTTALtype=gcn_lr=0.001_h=64_drop=0.2_JK=last_ms=0.8/epoch=12.checkpoint.pth.tar"
     elif args.modeltype == 'GNN':
          model = GNN(args)
          checkpoint_path = "../weights/GNN/year=2020_lr=0.001_h=64_drop=0.2_ms=0.8/epoch=17.checkpoint.pth.tar"
     elif args.modeltype == 'PreModel_v4':
          model = PreModel_v4(args)
          checkpoint_path = "../weights/PreModel_v4/year=2020_lr=0.001_h=64_drop=0.2_ms=0.8/epoch=10.checkpoint.pth.tar"
     elif args.modeltype == "GNN_v2":
          model = GNN_v2(args)
          checkpoint_path = "../weights/GNN_v2/year=2020_lr=0.001_h=64_drop=0.2_ms=0.8/epoch=6.checkpoint.pth.tar"


     model.load_state_dict(torch.load(checkpoint_path, map_location = "cuda:{}".format(args.cuda))['state_dict'])
     model.eval()
     model.cuda(device)

     logger.debug("model path: {}".format(checkpoint_path))

     if args.white_box:
          substitude_model = None
     else:
          substitude_model = BlackBox(args)
          substitude_model.load_state_dict(torch.load(checkpoint_path_sub, map_location="cuda:{}".format(args.cuda))['state_dict'], strict=False)
          substitude_model.eval()
          substitude_model.cuda(device)
          logger.debug("substitute model path: {}".format(checkpoint_path_sub))

     if args.modeltype == 'PreModel_v3':
          jsma = JSMA_c2(model=model, logger = logger, args = args, substitute_model=substitude_model)
     elif args.modeltype == 'GNN_VAE':
          jsma = JSMA_gnn_vae(model=model, logger = logger, args = args, substitute_model = substitude_model)
     elif args.modeltype == 'MsDroid':
          jsma = JSMA_msdroid(model=model, logger = logger, cuda = args.cuda, substitute_model=substitude_model)
     elif args.modeltype == 'GNN':
          jsma = JSMA(model=model, logger = logger, args = args, substitute_model=substitude_model)
     elif args.modeltype == 'PreModel_v4':
          jsma = JSMA_c2(model=model, logger = logger, args = args, substitute_model=substitude_model)
     elif args.modeltype == 'GNN_v2':
          jsma = JSMA_gnn_v2(model=model, logger = logger, args = args, substitute_model=substitude_model)


     jsma.attack()
