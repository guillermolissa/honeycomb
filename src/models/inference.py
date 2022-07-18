# inference.py
# desc: load model who has been build and score test (submit) dataset
# @autor: glissa
# created: 2021/01/07
import argparse 
import os
import logging
import warnings
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
import gc
import pickle
from datetime import datetime
from time import time
import config
import model_dispatcher
import metric_dispatcher
warnings.filterwarnings("ignore")
#from plot_model import plot_roc_cv


def get_next_filename(model):
    '''Gets the next numeric filename in a sequence.

    All files in the output directory must have the same name format,
    e.g. "txt1.txt".
    '''

    n = 0
    for f in os.listdir(config.SUBMIT_PATH):
        n = max(n, int(get_num_part(os.path.splitext(f)[0])))
    return f"submit_{model}_n{(n + 1)}.csv" 


def get_num_part(s):
    '''Get the numeric part of a string of the form "abc123".

    Quick and dirty implementation without using regex.'''

    for i in range(len(s)):
        if s[i:].isdigit():
            return s[i:]
    return ''





def load_model(modelname):

    # load the model from disk
    model = pickle.load(open(f"{config.MODEL_PATH}{modelname}.bin", 'rb'))

    return model

    pass



def submit(prediction, modelname, data):
    """
    This function should be defined by user 
    """

    data['prediction'] = prediction
    data.reset_index(inplace=True)
    submit = data[[config.KEY_ID,config.LABEL]]

    # get name of next submit file
    submit_filename = get_next_filename(modelname)

    submit.to_csv(f"{config.SUBMIT_PATH}{submit_filename}", index=False)

    if config.KAGGLE_SUBMIT:
        os.system(f"kaggle competitions submit -c {config.KAGGLE_PROJECT} -f {config.SUBMIT_PATH}{submit_filename} -m 'submit file: {submit_filename} from model: {model}'")
    pass




def run(modelname, input_file):
    logger = logging.getLogger(__name__)
    logger.info(f'INIT: predict {modelname} model')

        # load train and validation datasets
        test = load_data(input_file)   

        logger.info(f'RUN: loading model: {modelname}')
        # fetch the model from model_dispatcher
        model = load_model(modelname)

        logger.info(f'RUN: predic model: {modelname}')
        y_pred = []


        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(test)[:,1] # classification

        else:
            y_pred = model.predict(test) # regression



        
        logger.info(f'SAVE PREDICTION: {modelname}' )
        save_predic_proba(y_pred, modelname, test)

        
        logger.info(f'END: predict {model} model' )
       
    pass


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument( "--kind",   required=True, 
        help="Specify if problem will be a classification or regression. Values: 'classification' or 'regression'", type=str )
    parser.add_argument( "--model",  required=True, 
        help="Select kind of model to be used to score from model_dispatcher. Ex: 'rf' equal to Random Forest", type=str )
    
    parser.add_argument( "--input_file",   required=True, 
        help="Specify file name without extension who will be scored by model.", type=str)
    
    args = vars(parser.parse_args())
    

    assert args['kind'] in ['classification', 'regression'], f"'kind' should be 'classification' or 'regression'. {args['kind']} was provided"

    run(kind=args['kind'], model=args['model'], input_file=args['input_file'])