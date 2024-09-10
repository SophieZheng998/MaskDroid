python main.py --modeltype BlackBox --batch_size 32 --lr 1e-3 --blackboxtype PreModel_v3 --train_year all --need_pretrain --saveID MaskDroid

python main.py --modeltype BlackBox --batch_size 32 --lr 1e-3 --blackboxtype GNN_VAE --train_year all --need_pretrain --num_layers 1 --saveID FDVAE

python main.py --modeltype BlackBox --batch_size 32 --lr 1e-3 --blackboxtype MsDroid --train_year all --need_pretrain --saveID Msdroid