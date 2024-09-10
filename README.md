# MaskDroid
Official code of "MaskDroid: Robust Android Malware Detection with Masked Graph Representations" published on 39th IEEE/ACM International Conference on Automated Software Engineering (ASE '24)

## Overview
MaskDroid aims to build a powerful malware detector with remarkable robustness against adversarial attacks. 
 
## Installation

Main packages: torch==2.3.1

## Run DIR

To run effectiveness experiments:
```bash
python main.py --modeltype PreModel_v3 --batch_size 32 --lr 1e-3  --train_year all --need_pretrain --need_record --mask_rate 0.8
```

To run adversarial attack:
```bash
python model/attack.py --modeltype PreModel_v3 --white_box 
```
or for blackbox attack:

```bash
python model/attack.py --modeltype PreModel_v3
```

To run concept drift experiment:
```bash
python main.py --modeltype PreModel_v3 --concept_drift --sh --restore_epoch  19  --train_year 2019 --test_year 2020  --batch_size 32 --lr 1e-3  
```

Complete list of scripts are available in /scripts

