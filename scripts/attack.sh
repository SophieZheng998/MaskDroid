#python model/attack.py --modeltype PreModel_v3 --white_box --cuda 1

#python model/attack.py --modeltype PreModel_v3  --cuda 1

#python attack.py --modeltype GNN_VAE --white_box --num_layers 1

#python attack.py --modeltype GNN_VAE  --num_layers 1

#python attack.py --modeltype MsDroid --white_box 

#python attack.py --modeltype MsDroid 


python attack.py --modeltype PreModel_v3 --saveID rate=0_2 --ablation --white_box 

python attack.py --modeltype PreModel_v3 --saveID rate=0_5 --ablation --white_box --cuda 1

python attack.py --modeltype PreModel_v3 --saveID rate=0_8 --ablation --white_box --cuda 1

python attack.py --modeltype PreModel_v3 --saveID rate=0_9 --ablation --white_box --cuda 4

python attack.py --modeltype GNN  --ablation --white_box 

python attack.py --modeltype GNN_v2  --ablation --white_box --cuda 1

python attack.py --modeltype PreModel_v4  --ablation --white_box --cuda 1

