# kfolds.py
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

def make_folds(data, target='target', folds=3, task='classification', seed=47):
    """Split dataframe in k folds by target and add new variable kfold with k values correponding to each fold

    Args:
        data (Pandas dataframe): Dataframe to split into k folds
        target (str, optional): Target value from where the dataframe will be splited into k folds. Defaults to 'target'.
        folds (int, optional): Number of folds. Defaults to 3.
        task (str, optional): specifies what kind of problem we have. Only two values posible ['classification','regression']. Defaults to 'classification'.
        seed (int, optional): Seed value to random select . Defaults to 47.

    Returns:
        [type]: Original dataframe with a new variable kfold who content the k fold value to split the dataframe.
    """

    assert task in ['classification', 'regression'], f"'task' should be 'classification' or 'regression'. {task} was provided"

    if task == 'classification':
    
        # Training data is in a csv file called train.csv df = pd.read_csv("train.csv")
        # we create a new column called kfold and fill it with -1
        data["kfold"] = -1

        # the next step is to randomize the rows of the data
        data = data.sample(frac=1).reset_index(drop=True) # fetch targets
        y = data.loc[:, target].values
        
        # initiate the kfold class from model_selection module
        kf = model_selection.StratifiedKFold(n_splits=folds)

        # fill the new kfold column
        for f, (t_, v_) in enumerate(kf.split(X=data, y=y)): 
            data.loc[v_, 'kfold'] = f

    elif task == 'regression':
        # we create a new column called kfold and fill it with -1 data["kfold"] = -1
        # the next step is to randomize the rows of the data
        data = data.sample(frac=1).reset_index(drop=True)

        # calculate the number of bins by Sturge's rule # I take the floor of the value, you can also # just round it
        num_bins = np.floor(1 + np.log2(len(data)))
        
        # bin targets
        data.loc[:, "bins"] = pd.cut(
            data[target], bins=num_bins, labels=False
            )
        
        # initiate the kfold class from model_selection module
        kf = model_selection.StratifiedKFold(n_splits=folds)

        # fill the new kfold column
        # note that, instead of targets, we use bins!
        for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
            data.loc[v_, 'kfold'] = f
        
        # drop the bins column
        data = data.drop("bins", axis=1) # return dataframe with folds return data

    # return the new dataframe with kfold column
    return data
    
    

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument( "--input_file",   required=True, 
        help="Specify file from features folder to be splited without extension.", type=str )
    parser.add_argument( "--kfold",   required=True, 
        help="Specify the number of K folds in which the dataset will be splited.", type=int)
    
    args = vars(parser.parse_args())
    
    logger = logging.getLogger(__name__)
    logger.info(f"INIT: creating {args['kfold']} folds on {args['input_file']}")

    # LOAD DATASET
    data = load_data(args['input_file'])

    # FOLD
    data = make_folds(data, target=config.LABEL, folds=args['kfold'], task=config.TASK, seed=config.SEED)

    # SAVE DATASET
    save_data(args['input_file'], data)
    

    logger.info('END: folds created successfully.')