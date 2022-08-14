# hold_out.py
#!/usr/bin/env python
# coding: utf-8

# Import libraries
import logging
import argparse 
import pandas as pd
import numpy as np
import random
from sklearn import model_selection
import config
from data.data_io import load_data, save_data
import os 


def make_hold_out(data,  target='target', test_size=0.3, seed=47):

    y = data.loc[:, target]

    X_train, X_test= model_selection.train_test_split(data, test_size=test_size, random_state=seed, stratify=y)

    X_train.loc[:, "kfold"] = 0
    X_test.loc[:, "kfold"]  = 1

    return pd.concat([X_train,X_test],axis=0)




if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument( "--input_file",   required=True, 
        help="Specify file from features folder to be splited without extension.", type=str )
    parser.add_argument( "--test_size",   required=False, 
        help="Specify he proportion of the dataset to include in the test split. It should be between 0.0 and 1.0 ", type=float)
    
    args = vars(parser.parse_args())
    
    logger = logging.getLogger(__name__)
    logger.info(f"INIT: creating hold out in {args['input_file']}")


    test_size = config.TEST_SIZE

    if args['test_size'] is not None:
        test_size = args['test_size']


    # LOAD DATASET
    data = load_data(args['input_file'])

    # FOLD
    data = make_hold_out(data, target=config.LABEL, test_size=test_size , seed=config.SEED)
    train_df = data[data["kfold"]==0].drop("kfold",axis=1)
    test_df = data[data["kfold"]==1].drop("kfold",axis=1)

    # SAVE DATASET
    save_data(args['input_file'], data)

    

    logger.info('END: hould out created successfully.')