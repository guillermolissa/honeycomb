# train.py
# desc: train and validate model using CV
# @autor: glissa
# created: 2021/01/06
import argparse 
import logging
from unittest import result
import warnings
import pandas as pd
import numpy as np
import os
import joblib
import gc
import pickle
from datetime import datetime
from time import time
from config import Config as config
import model_dispatcher
import metric_dispatcher
from data.data_io import load_data
warnings.filterwarnings("ignore")




def train_eval(model, df, metricfun):
    

    folds =-1

    if 'kfold' in df.columns:
        folds = df.kfold.max()

    if folds != -1:

        print(f"Training CV Model='{type(model).__name__}'")


        cv_val_result = []
        for fold in range(folds + 1):
            # training data is where kfold is not equal to provided fold 
            # also, note that we reset the index
            df_train = df[df.kfold != fold].reset_index(drop=True)

            # validation data is where kfold is equal to provided fold
            df_valid = df[df.kfold == fold].reset_index(drop=True) # drop the label and kfold column from dataframe and convert it to
            
            # a numpy array by using .values.
            # target is label column in the dataframe
            x_train = df_train.drop([config.LABEL, 'kfold'], axis=1).values 
            y_train = df_train[config.LABEL].values
            
            # similarly, for validation, we have
            x_valid = df_valid.drop([config.LABEL, 'kfold'], axis=1).values 
            y_valid = df_valid[config.LABEL].values

            
            # fit the model on training data
            model.fit(x_train, y_train)

            # create predictions for validation samples
            preds = model.predict(x_valid)

            # calculate & print metric            
            result = metricfun(y_valid, preds) 

            print(f"Fold={fold}, {metricfun.__name__}={result}")

            cv_val_result.append(result)

            # calling garbage collector
            gc.collect()
            

            # save the model fold model
            joblib.dump( model, config.MODEL_OUTPUT + f"{type(model).__name__.lower()}_{fold}.bin") 

            
        print(f'Result: CV {type(model).__name__} - {metricfun.__name__} - mean=%.3f - std=(%.3f)' % (np.mean(cv_val_result), np.std(cv_val_result)))
    
    else:
        # a numpy array by using .values.
        # target is label column in the dataframe
        x_train = df.drop(config.LABEL, axis=1).values 
        y_train = df[config.LABEL].values
            
        

        print(f"Training Model = '{type(model).__name__}'")


        # fit the model on training data
        model.fit(x_train, y_train)
        

        # save the model
        joblib.dump( model, config.MODEL_OUTPUT + f"{type(model).__name__.lower()}.bin") 



if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument( "--model",  required=True, 
        help="Select algorithm to train a model from model_dispatcher. Ex: 'rf' equal to Random Forest", type=str )
    
    
    parser.add_argument( "--file_name",  required=True, 
        help="Select file or dataset to train the model. ", type=str )

    parser.add_argument( "--metric", 
        help="Metric that will be used to validate the performance of the model. Values must be provided from metric_dispatcher. Ex: 'accuracy'", required=True, type=str )
    
    
    args = vars(parser.parse_args())
    

    assert args['model'] in model_dispatcher.models.keys(), f"'{args['model']}' not found in model dispatcher. Try with {list(model_dispatcher.models.keys())}."
    assert args['metric'] in metric_dispatcher.metrics_score.keys(), f"'{args['metric']}' not found in metric dispatcher. Try with {list(metric_dispatcher.metrics_score.keys())}."


    logger = logging.getLogger(__name__)
    logger.info(f"INIT: train  Model={args['model']}")


    
    logger.info(f'RUN: loading data ')
    
    # read the training data with folds 
    df = load_data(args['file_name'])


    # fetch the model from model_dispatcher
    model = model_dispatcher.models[args['model']]

    metricfun = metric_dispatcher.metrics_score[args['metric']]


    logger.info(f"RUN: training Model = {type(model).__name__}")


    train_eval(model, df, metricfun)

    logger.info(f"END: train  Model={args['model']}")


