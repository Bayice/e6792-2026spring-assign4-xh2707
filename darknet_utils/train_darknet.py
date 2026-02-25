"""
This file is used for initiating darknet training. You aren't required to use it.

E6792 Spring 2026

YOU DO NOT NEED TO MODIFY THIS FUNCTION TO COMPLETE THE ASSIGNMENT
"""

import os

def train_darknet(data_cfg_path, model_cfg_path, initial_weights):
    print(os.getcwd())
    os.system('sudo ./darknet detector train {} {} {} -dont_show -map &'.format(data_cfg_path,
                                                                                model_cfg_path,
                                                                                initial_weights))
    model_name = model_cfg_path.split('/')[-1].split('.')[0]
    dataset_name = data_cfg_path.split('/')[-1].split('.')[0]
    print("Darknet training started for model {} with dataset {}.".format(model_name, dataset_name))
