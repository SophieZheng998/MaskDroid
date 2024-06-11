import os
import random
import numpy as np
import pandas as pd
import torch

from argparse import Namespace
from logging import Logger

def ensure_dir(file_path: str):
    """
    Ensure the directory of the file exists.
    :param file_path: The path to the file.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def ensure_file(file_path:str):
    """
    Ensure the file exists.
    :param file_path: The path to the file.
    """
    if not os.path.exists(file_path):
        open(file_path, 'w').close()

def check_args(args: Namespace, required_args: list):
    """
    Check if the required arguments are set.
    :param args: The arguments.
    :param required_args: The required arguments.
    :return: The other parameters.
    """
    for arg in required_args:
        if not hasattr(args, arg):
            raise Exception("The argument %s is not set." % arg)
        if getattr(args, arg) is None:
            raise Exception("The argument %s is not set." % arg)
    
    other_params = {}
    args_dict = vars(args)
    for arg, value in args_dict.items():
        if arg not in required_args:
            other_params[arg] = value
    
    return other_params

def read_feature_dirs(path: str):
    """
    Read the feature directories from feature_dir_path.txt
    :param path: The path to feature_dir_path.txt
    :return: The feature directories.
    """
    if not os.path.exists(path):
        raise Exception("The path %s does not exist." % path)
    with open(path, 'r') as f:
        feature_dirs = f.read().splitlines()
    
    return feature_dirs

def set_seed(seed: int):
    """
    Set the seed for the random number generator.
    :param seed: The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def files_exist(file_paths: list):
    """
    Check if the files exist.
    :param file_paths: The path to the files.
    :return: True if all the files exist, False otherwise.
    """
    for file_path in file_paths:
        if not os.path.exists(file_path):
            return False
    return True

# Helper function for the data splitting
def get_apk_names(apk_path: str):
    """
    Get the APK names.
    :param apk_path: The path to the APKs.
    :return: The APK names.
    """
    if not os.path.exists(apk_path):
        raise Exception("The path %s does not exist." % apk_path)
    with open(apk_path, 'r') as f:
        apk_names = f.read().splitlines()
    
    return apk_names

def downsampling(apks: list, num: int):
    """
    Downsample the APKs.
    :param apks: The APKs.
    :param num: The number of APKs to downsample to.
    """
    if len(apks) <= num:
        return apks
    else:
        return random.sample(apks, num)

def sampling_apks(benign_apks: list, malware_apks: list, bm_ratio: float):
    """
    Sample the benign and malware APKs.
    """
    malware_ratio = bm_ratio
    assert malware_ratio >= 0 and malware_ratio <= 1
    
    num_goodware = int(len(malware_apks) / malware_ratio) - len(malware_apks)
    if len(benign_apks) >= num_goodware:
        sampled_benign_apks = random.sample(benign_apks, num_goodware)
        sampled_malware_apks = malware_apks
    else:
        num_maleware = int(len(benign_apks) / (1 - malware_ratio)) - len(benign_apks)
        if num_maleware > len(malware_apks):
            raise Exception("The number of malware APKs is not enough.")
        sampled_benign_apks = benign_apks
        sampled_malware_apks = random.sample(malware_apks, num_maleware)
    
    return sampled_benign_apks, sampled_malware_apks

def train_test_split(benign_apks: list, malware_apks: list, ratio: str):
    """
    Split the benign and malware APKs into training, validation, and testing sets.
    """
    ratios = ratio.split(':')
    assert len(ratios) == 3
    ratios = [int(r) for r in ratios]
    # assert sum(ratios) == 10
    total = sum(ratios)

    # Randomly shuffle the benign and malware APKs
    random.shuffle(benign_apks)
    random.shuffle(malware_apks)
    # Get the number of benign and malware APKs
    num_benign_apks = len(benign_apks)
    num_malware_apks = len(malware_apks)

    # Split benign APKs
    num_benign_train = int(num_benign_apks * ratios[0] / total)
    num_benign_val = int(num_benign_apks * ratios[1] / total)
    benign_train = benign_apks[:num_benign_train]
    benign_val = benign_apks[num_benign_train:num_benign_train + num_benign_val]
    benign_test = benign_apks[num_benign_train + num_benign_val:]

    # Split malware APKs
    num_malware_train = int(num_malware_apks * ratios[0] / total)
    num_malware_val = int(num_malware_apks * ratios[1] / total)

    malware_train = malware_apks[:num_malware_train]
    malware_val = malware_apks[num_malware_train:num_malware_train + num_malware_val]
    malware_test = malware_apks[num_malware_train + num_malware_val:]

    # Merge benign and malware APKs
    train = benign_train + malware_train
    train_sample_50 = random.sample(benign_train, int(len(benign_train) / 2)) + random.sample(malware_train, int(len(malware_train) / 2))
    train_sample_10 = random.sample(benign_train, int(len(benign_train) / 10)) + random.sample(malware_train, int(len(malware_train) / 10))
    train_sample_5 = random.sample(benign_train, int(len(benign_train) / 20)) + random.sample(malware_train, int(len(malware_train) / 20))

    val = benign_val + malware_val
    test = benign_test + malware_test

    return train, val, test, train_sample_50, train_sample_10, train_sample_5

def write_data_distribution(train: list, val: list, test: list, output_path: str):
    """
    Write training, validation, and testing data distributions.
    :param train: The training data.
    :param val: The validation data.
    :param test: The testing data.
    """
    ensure_dir(output_path)
    dic = {
        'train': train,
        'val': val,
        'test': test
    }
    with open(output_path, 'w') as f:
        f.write(str(dic))

def print_dataset_statistics(base_dir: str, logger: Logger):
    """
    Print the dataset statistics.
    """
    # 2011-2020 / 1-12
    years = [str(y) for y in range(2011, 2021)]
    months = [str(m) for m in range(1, 13)]

    # Get the number of benign and malware APKs
    benign_all_num = 0
    malware_all_num = 0
    for year in years:
        benign_apks_num = 0
        malware_apks_num = 0
        for month in months:
            benign_apks = get_apk_names(os.path.join(base_dir, 'Benign_'+year+'_'+month+'.txt'))
            malware_apks = get_apk_names(os.path.join(base_dir, 'Malware_'+year+'_'+month+'.txt'))
            benign_apks_num += len(benign_apks)
            malware_apks_num += len(malware_apks)
        total_apks_num = benign_apks_num + malware_apks_num
        malware_ratio = malware_apks_num / total_apks_num
        logger.info("Year: %s, # of total APKs: %d, # of benign APKs: %d, # of malware APKs: %d, malware ratio: %.2f%%" % (year, total_apks_num, benign_apks_num, malware_apks_num, malware_ratio * 100))
        benign_all_num += benign_apks_num
        malware_all_num += malware_apks_num
    total_all_num = benign_all_num + malware_all_num
    malware_all_ratio = malware_all_num / total_all_num
    logger.debug("Total, # of total APKs: %d, # of benign APKs: %d, # of malware APKs: %d, malware ratio: %.2f%%" % (total_all_num, benign_all_num, malware_all_num, malware_all_ratio * 100))

def calculate_avg_apk_size(base_dir: str, apk_info: str, logger: Logger):
    """
    Calculate the average size of apps.
    """
    # 2011-2020 / 1-12
    years = [str(y) for y in range(2011, 2021)]
    months = [str(m) for m in range(1, 13)]

    # Read apk info from csv file
    apk_info_df = pd.read_csv(apk_info)
    apk_info_df = apk_info_df[['sha256', 'apk_size']]

    # Get the name of the APKs
    for year in years:
        apk_names = []
        for month in months:
            apk_names += get_apk_names(os.path.join(base_dir, 'Benign_'+year+'_'+month+'.txt'))
            apk_names += get_apk_names(os.path.join(base_dir, 'Malware_'+year+'_'+month+'.txt'))
        # Get the size of the APKs
        selected_apk_info_df = apk_info_df.loc[apk_info_df['sha256'].isin(apk_names), 'apk_size']
        avg_apk_size = selected_apk_info_df.mean()
        logger.info("Year: %s, average APK size: %.4f" % (year, avg_apk_size/(1024*1024)))