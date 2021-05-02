# make_train_test_dataset.py
#!/usr/bin/env python
# coding: utf-8

# Import libraries
import logging
import argparse 
import pandas as pd
import numpy as np
import random
import config
from sklearn import model_selection




def make_kfold_classif(data, target='target', folds=3, seed=47):
    
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

    # return the new dataframe with kfold column
    return data



def make_kfold_regression(data,  target='target', folds=3, seed=47):
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
    

    return data
    
    

def make_hold_out(data,  target='target', test_size=0.3, seed=47):

    data["kfold"]=-1
    y = data.loc[:, target]

    X_train, X_test= model_selection.train_test_split(data["kfold"], test_size=test_size, random_state=seed, stratify=y)

    X_train.loc[:, "kfold"] = 0
    X_test.loc[:, "kfold"]  = 1

    return pd.concat([X_train,X_test],axis=0)