from sys import exit
from utils import *
from parse import parse_args


if __name__ == '__main__':
    args = parse_args()
    seed_torch(args.seed) 

    #try:
    exec('from model.'+ args.modeltype + ' import ' + args.modeltype + '_RUN') # load the model
    #except:
    #print('Model %s not implemented!' % (args.modeltype))
    #exit(1)
        
    RUN = eval(args.modeltype + '_RUN(args)')

    RUN.execute() 
    
