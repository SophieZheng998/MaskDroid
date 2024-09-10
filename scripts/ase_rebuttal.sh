python main.py --modeltype PreModel_v3 --concept_drift --train_year 2016-2019 --test_year 2020  --batch_size 32 --lr 1e-3  --mask_rate 0.4 --need_pretrain --need_record --num_layers 1 

python main.py --modeltype MsDroid --concept_drift  --train_year 2016-2019  --test_year 2020 --batch_size 1 --lr 1e-3   --need_pretrain --num_layers 1 --need_record --cuda 7

python main.py --modeltype GNN_VAE --concept_drift --train_year 2016-2019 --test_year 2020 --batch_size 32 --lr 1e-3  --need_pretrain --num_layers 1  --need_record --cuda 0

python Malscan.py --modeltype knn  --concept_drift  --train_year 2016-2019  --test_year 2020 --need_record

python Mamadroid.py --modeltype rf --concept_drift --train_year 2016-2019  --test_year 2020 --need_record

python Drebin.py --modeltype svm  --concept_drift --train_year 2016-2019  --test_year 2020 --need_record

python main.py --modeltype BlackBox --batch_size 32 --lr 1e-3 --blackboxtype GNN_VAE --train_year 2016-2019 --test_year 2020 --need_pretrain --num_layers 1 --saveID FDVAE_rebuttal

python main.py --modeltype BlackBox --batch_size 32 --lr 1e-3 --blackboxtype PreModel_v3 --train_year 2016-2019 --need_pretrain --saveID MaskDroid_rebuttal --cuda 2 

python main.py --modeltype PreModel_v3 --sh --restore_epoch 7 --concept_drift --train_year 2016-2019 --test_year 2016-2019  --batch_size 32 --lr 1e-3  --mask_rate 0.8  --need_record --num_layers 1  --cuda 3

python main.py --modeltype GNN_VAE --sh --restore_epoch 1 --concept_drift --train_year 2016-2019 --test_year  2016-2019 --batch_size 32 --lr 1e-3  --need_pretrain --num_layers 1  --need_record --cuda 0

python main.py --modeltype MsDroid --concept_drift --sh  --train_year 2016-2019 --test_year  2016-2019 --batch_size 1 --lr 1e-3  

python attack.py --model PreModel_v3 --white_box --cuda 0 --saveID rebuttal

python attack.py --model PreModel_v3 --cuda 1 --saveID rebuttal

python attack.py --modeltype GNN_VAE --white_box --num_layers 1 --cuda 1 --saveID rebuttal_iter10

python attack.py --modeltype GNN_VAE --num_layers 1 --cuda 3 --saveID rebuttal

python attack.py --modeltype MsDroid --white_box --cuda 2 --saveID rebuttal

python main.py --modeltype BlackBox --batch_size 32 --lr 1e-3 --blackboxtype MsDroid --train_year all --need_pretrain --saveID Msdroid_rebuttal


python main.py --modeltype PreModel_v3 --batch_size 32 --lr 1e-3  --train_year 2020 --need_pretrain --mask_rate 0.1 --saveID rate=0_1 --cuda 0 

python main.py --modeltype PreModel_v3 --batch_size 32 --lr 1e-3  --train_year 2020 --need_pretrain --mask_rate 0.3 --saveID rate=0_3 --cuda 1

python main.py --modeltype PreModel_v3 --batch_size 32 --lr 1e-3  --train_year 2020 --need_pretrain --mask_rate 0.4 --saveID rate=0_4 --cuda 0 

python main.py --modeltype PreModel_v3 --batch_size 32 --lr 1e-3  --train_year 2020 --need_pretrain --mask_rate 0.6 --saveID rate=0_6 --cuda 1

python main.py --modeltype PreModel_v3 --batch_size 32 --lr 1e-3  --train_year 2020 --need_pretrain --mask_rate 0.7 --saveID rate=0_7 --cuda 0

python main.py --modeltype PreModel_v3 --batch_size 32 --lr 1e-3  --train_year 2020 --need_pretrain --mask_rate 0.9 --saveID rate=0_9  --cuda 1


python attack.py --modeltype PreModel_v3 --saveID rate=0_1 --ablation --white_box --cuda 0 

python attack.py --modeltype PreModel_v3 --saveID rate=0_3 --ablation --white_box --cuda 1 

python attack.py --modeltype PreModel_v3 --saveID rate=0_4 --ablation --white_box --cuda 2 

python attack.py --modeltype PreModel_v3 --saveID rate=0_6 --ablation --white_box --cuda 3 

python attack.py --modeltype PreModel_v3 --saveID rate=0_7 --ablation --white_box --cuda 0 