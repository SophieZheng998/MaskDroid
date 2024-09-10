import argparse


def parse_args():
    parser = argparse.ArgumentParser()
 
    # execution
    parser.add_argument('--seed', type=int, default=101,
                        help='Random seed.')
    parser.add_argument('--saveID', type=str, default="",
                        help='Specify model save path.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Specify which gpu to use.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epoch', type=int, default=20,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--reg', type=float, default=1e-4,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping point.')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='log\'s interval epoch while training')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--modeltype', type=str, default= 'PreModel_v3',
                        help='Specify model save path.')
    parser.add_argument('--infonce', type=int, default=0,
                        help='whether to use infonce loss or not')
    parser.add_argument('--clear_checkpoints', action="store_true",
                        help='Whether clear the earlier checkpoints.')
    parser.add_argument('--max2keep', type=int, default=10,
                        help='max checkpoints to keep')
    
    # model

    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GCN layers')
    parser.add_argument('--in_channels', type=int, default=492,
                        help='Number of GCN layers')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Number of hidden features in GCN layers')
    parser.add_argument('--mid_channels', type=int, default=64,
                        help='Number of hidden features in the last layer of readout')
    parser.add_argument('--out_channels', type=int, default=2,
                        help='Number of output classes')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='drop out ratio in training')
    parser.add_argument('--gnn_type', type=str, default="gcn",
                        help='[gin, sage, gcn, gat]')
    parser.add_argument('--JK', type=str, default="last",
                        help='[last, sum, concat]')
    
    # contrastive loss
    parser.add_argument('--contrastive', action="store_true",
                        help='Whether to add contrastive loss.')
    parser.add_argument('--eps', type=float, default=0.1,
                            help='Noise rate')
    parser.add_argument('--lambda_cl', type=float, default=1,
                                help='Rate of contrastive loss')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='temperature parameter')

    # diffusion 
    parser.add_argument('--alpha', type=float, default=0.05,
                            help='alpha')
    parser.add_argument('--k', type=int, default=128,
                            help='alpha')
    parser.add_argument('--diffusion', action="store_true",
                        help='Whether add GDN.')

    # GraphMAE
    parser.add_argument('--mask_rate', type=float, default=0.8,
                                help='Rate of contrastive loss')
    parser.add_argument('--test_mask_rate', type=float, default=0.5,
                                help='Rate of contrastive loss')
    parser.add_argument('--in_channels_cp', type=int, default=64,
                        help='Number of hidden features in GCN layers')
    parser.add_argument('--lambda_rec', type=float, default=1,
                                help='Rate of contrastive loss')
    parser.add_argument('--need_pretrain', action="store_true",
                        help='first pretrain the encoder-decoder')
    parser.add_argument('--thres', type=float, default=0.2,
                                help='threshold to check if the sample is benign')
    
    parser.add_argument('--margin', type=float, default=0.1,
                                help='margin')

    # FD-VAE
    parser.add_argument('--lambda1', type=float, default=1)
    parser.add_argument('--lambda2', type=float, default=1)
    parser.add_argument('--lambda3', type=float, default=1)
    

        
    # blackbox
    parser.add_argument('--blackboxtype', type=str, default= 'GNN',
                        help='Specify model save path.')
    
    parser.add_argument('--white_box', action="store_true",
                        help='white_box or blackbox for baseline model attack.')

    # test bias
    parser.add_argument('--test_bias', action="store_true",
                        help='in attack, test the temporal bias examples.')

    # dataset

    parser.add_argument('--train_year', type=str, default="2016",
                        help='[2016, 2017, 2018, 2019, 2020, all]')


    parser.add_argument('--concept_drift', action="store_true",
                        help='test concept drift or not')

    parser.add_argument('--test_year', type=str, default="2017",
                        help='[2017, 2018, 2019, 2020]')

    parser.add_argument('--need_record', action="store_true",
                        help='record file names of malware precisely detected')

    parser.add_argument('--train_flag', type=str, default ="train",
                        help='[train, test]')

    parser.add_argument('--ablation', action="store_true",
                        help='whether the attack is for ablation study or not')

    # running sh
    parser.add_argument("--sh", action="store_true",
                        help = "whether need to restore from pretrained checkpoints")

    parser.add_argument("--restore_epoch", type=int, default="0")


    args = parser.parse_args()

    
    return args