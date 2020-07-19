# make_train_test_dataset.py
#!/usr/bin/env python
# coding: utf-8

# Import libraries
import argparse 
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split  #Additional scklearn functions
import config


# DEF FUNCTIONS

# LOAD DATASET
def load_data(file_name):
    X = pd.read_pickle(config.INPUT_PATH + f'features/{file_name}.pkl').drop(config.target, axis=1)
    X.columns = [str(c) for c in X.columns]

    y = pd.read_pickle(config.INPUT_PATH + f'features/{file_name}.pkl')[config.target]
    return X,y

# SAVE TRAIN & TEST
def save_and_split_data(X,y):
    # SPLIT IN TRAIN/TEST
    # set the seed of random number generator, which is useful for creating simulations 
    # or random objects that can be reproduced.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y)

    X_train.shape
    X_test.shape
    y_train.shape
    y_test.shape

    X_train.to_pickle(config.OUTPUT_PATH + 'X_train.pkl')
    del X_train

    X_test.to_pickle(config.OUTPUT_PATH + 'X_test.pkl')
    del X_test

    y_train.to_pickle(config.OUTPUT_PATH + 'y_train.pkl')
    del y_train

    y_test.to_pickle(config.OUTPUT_PATH + 'y_test.pkl')
    del y_test

    return 0
    


def run(file_name):
    # LOAD DATASET
    X,y = load_data(file_name)

    save_and_split_data(X,y) 
    
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--file",   required=True, 
        help="Specify file from features folder to be splited without extension.", type=str )
    
    args = vars(parser.parse_args())
    
    run(file_name=args['file'])