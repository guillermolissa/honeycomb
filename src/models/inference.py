# inference.py
# desc: load model who has been build and score test (submit) dataset
# @autor: glissa
# created: 2021/01/07
import argparse 
import os
import logging
import warnings
import pandas as pd
import gc
import joblib
from time import time
from config import Config as config
from data.data_io import load_data
from config import Config as config
import model_dispatcher as model_dispatcher
warnings.filterwarnings("ignore")
#from plot_model import plot_roc_cv


def get_next_filename(filename):
    '''Gets the next numeric filename in a sequence.

    All files in the output directory must have the same name format,
    e.g. "txt1.txt".
    '''

    n = 0
    for f in os.listdir(config.SUBMIT_PATH):
        n = max(n, int(get_num_part(os.path.splitext(f)[0])))
    return f"{filename}_n{(n + 1)}.csv" 


def get_num_part(s):
    '''Get the numeric part of a string of the form "abc123".

    Quick and dirty implementation without using regex.'''

    for i in range(len(s)):
        if s[i:].isdigit():
            return s[i:]
    return ''





def load_model(modelname):

    # load the model from disk
    model = joblib.load(open(f"{config.MODEL_PATH}{modelname}.bin", 'rb'))

    return model

    pass



def submit(filename, submit):
    """
    This function should be defined by user 
    """

    submit.reset_index(inplace=True)

    # get name of next submit file
    submit_filename = get_next_filename(filename)

    submit.to_csv(f"{config.SUBMIT_PATH + submit_filename}", index=False)

    if config.KAGGLE_SUBMIT:
        os.system(f"kaggle competitions submit -c {config.KAGGLE_PROJECT} -f {config.SUBMIT_PATH + submit_filename} -m 'submit file: {submit_filename}")
    pass



def predict(model, test):
    y_pred = []
    features = test.columns

    if hasattr(model, 'predict_proba'):
        y_pred = model.predict_proba(test)[:,1] # classification

    else:
        y_pred = model.predict(test) # regression

    test["prediction"] = y_pred 

    test.drop(features, axis=1, inplace=True)
       
    return test


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument( "--model",  required=True, 
        help="Select kind of model to be used to score from model_dispatcher. Ex: 'rf' equal to Random Forest", type=str )
    
    parser.add_argument( "--input_file",   required=True, 
        help="Specify file name without extension who will be scored by model.", type=str)
    
    args = vars(parser.parse_args())
        
    modelname = args['model']

    logger = logging.getLogger(__name__)
    logger.info(f"INIT: predict {modelname} model")

    
    logger.info(f"RUN: loading {args['input_file']}")

    # load train and validation datasets
    test_df = load_data(config.INPUT_PATH + args['input_file'])   
    test_df.set_index(config.KEY_ID, inplace=True)
    test_df = test_df[['limit_bal', 'sex', 'education', 'marriage', 'age', 'pay_0',
       'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6', 'bill_amt2', 'bill_amt4',
       'bill_amt6', 'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5',
       'pay_amt6', 'pastpayment_acum', 'bill_amount_acum', 'tend_bill_amt5',
       'tend_bill_amt4', 'tend_bill_amt3', 'tend_bill_amt2', 'tend_bill_amt1',
       'payment_amount_acum', 'tend_pay_amt5', 'tend_pay_amt4',
       'tend_pay_amt3', 'tend_pay_amt2', 'tend_pay_amt1']]


    logger.info(f"RUN: loading model: {modelname}")
        
    # fetch the model from model_dispatcher
    model = load_model(modelname)

    logger.info(f"RUN: predic model: {type(model).__name__}")

    submit_df = predict(model, test_df)


    logger.info(f"SAVE PREDICTION: {modelname}")

    submit("submission", submit_df)
        
    logger.info(f'END: predict {modelname} model' )