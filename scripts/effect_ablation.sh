python main.py --modeltype PreModel_v3 --batch_size 32 --lr 1e-3  --train_year 2020 --need_pretrain --mask_rate 0.2 --saveID rate=0_2

python main.py --modeltype PreModel_v3 --batch_size 32 --lr 1e-3  --train_year 2020 --need_pretrain --mask_rate 0.5 --saveID rate=0_5

python main.py --modeltype PreModel_v3 --batch_size 32 --lr 1e-3  --train_year 2020 --need_pretrain --mask_rate 0.9 --saveID rate=0_9

python main.py --modeltype GNN --batch_size 32 --lr 1e-3  --train_year all --need_pretrain 

python main.py --modeltype GNN_v2 --batch_size 32 --lr 1e-3  --train_year 2020 --need_pretrain 

python main.py --modeltype PreModel_v4 --batch_size 32 --lr 1e-3  --train_year 2020 --need_pretrain 