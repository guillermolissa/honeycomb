# train_model.py
import argparse
import os
import logging
import logging.handlers
import warnings
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_rand
import gc
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from time import time
import models.config as config
import models.hyperparams_dispatcher as hyperparams_dispatcher
import models.model_dispatcher as model_dispatcher
import models.metric_dispatcher as metric_dispatcher
warnings.filterwarnings("ignore")


# load datasets
def load_data(kind):
    
    if kind == 'classification':
        # read the training data
        X_train =  pickle.load(open( f'{config.INPUT_PATH}X_train.pkl', "rb" )) 
        y_train = pickle.load(open( f'{config.INPUT_PATH}y_train.pkl', "rb" ))

        # read test data
        X_test = pickle.load(open( f'{config.INPUT_PATH}X_test.pkl', "rb" )) 
        y_test = pickle.load(open( f'{config.INPUT_PATH}y_test.pkl', "rb" )) 

        return X_train, y_train, X_test, y_test

    else :
        # kind equal to regression
        
        # read the training data
        X_train =  pickle.load(open( f'{config.INPUT_PATH}X_train.pkl', "rb" )) 
        y_train = pickle.load(open( f'{config.INPUT_PATH}y_train.pkl', "rb" ))

        # read validation data
        X_val = pickle.load(open( f'{config.INPUT_PATH}X_val.pkl', "rb" )) 
        y_val = pickle.load(open( f'{config.INPUT_PATH}y_val.pkl', "rb" )) 


        # read test data
        X_test = pickle.load(open( f'{config.INPUT_PATH}X_test.pkl', "rb" )) 
        y_test = pickle.load(open( f'{config.INPUT_PATH}y_test.pkl', "rb" )) 

        return X_train, y_train, X_val, y_val, X_test, y_test


def test_best_model(model, metric, X_test, y_test):

    metric_function = metric_dispatcher.metrics_score[metric]

    return metric_function(y_test,model.predict_proba(X_test)[:,1])
    
    pass


def save_hypermaters(rsearch, model_name):

    cv_results = pd.DataFrame(rsearch.cv_results_)

    cv_results.to_csv(f'{config.MODEL_PATH}/{model_name}_hyperparams_rsearch.csv',sep=';',index=False)

    np.save(f'{config.MODEL_PATH}/{model_name}_bestparams_rsearch.npy', rsearch.best_params_)

    np.save(f'{config.MODEL_PATH}/{model_name}_bestestimator_rsearch.npy', rsearch.best_estimator_)


    pass 


def run(kind, model, folds, metric):
    logger = logging.getLogger(__name__)
    handler = logging.handlers.SysLogHandler(address = '/dev/log')
    logger.addHandler(handler)

    logger.info(f'INIT: train CV {kind} model')

    if kind == 'classification':
        # load train and validation datasets
        X_train, y_train, X_test, y_test = load_data(kind)   

        # fetch the model from model_dispatcher and hyperparameters
        clf = model_dispatcher.models[model]
        hyperparameters = hyperparams_dispatcher[model] 

        metricfun = metric_dispatcher.metrics_score[metric]
        
        logger.info(f'RUN: Randomized search CV model - {model}')

        # Random search of parameters, using CV fold cross validation, 
        # search across NITER different combinations, and use all available cores
        rsearch = RandomizedSearchCV(estimator = clf, param_distributions = hyperparameters, 
                                        scoring=metricfun, n_iter = config.N_ESTIMATORS, 
                                        cv = folds, verbose=config.VERBOSE, random_state=config.SEED, n_jobs = config.NJOBS) 


        # Fit the random search model
        start = time()
        rsearch.fit(X_train, y_train)
        logger.info(f"RUN: Randomized search CV model - {model} took { ((time() - start))} seconds for { config.N_ESTIMATORS} candidates parameter settings.")
        
        logger.info("RUN: Best estimator found: ")
        logger.info(f"{rsearch.best_estimator_}")

        logger.info("RUN: Best parameters found: ")
        logger.info(f"{rsearch.best_params_}")


        

        # get test score result
        logger.info(f'RUN: CV result - {metric} - mean {rsearch.best_score_}')

        test_result = test_best_model(rsearch, metric, X_test, y_test)    


        logger.info(f'RUN: test CV result - {metric} - {test_result} ')


        # saving final models
        
        logger.info(f'RUN: save hyperparameters model - {model}' )
        
        save_hypermaters(rsearch, model) 

        logger.info(f'END: Randomized search CV model - {model}' )
       

        pass
    else:
        # kind equal to regression

        pass


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument( "--kind",   required=True, 
        help="Specify if problem will be a classification or regression. Values: 'classification' or 'regression'", type=str )
    parser.add_argument( "--model",	 required=True, 
        help="Select kind of model to be used to train from model_dispatcher. Ex: 'rf' equal to Random Forest", type=str )
    parser.add_argument( "--folds",  
        help="Number of folds to be used in order to implement Cross Validation. This arg must be provided only if you are using classification, for regression problems, folds must be done using time split methods.", required=False, type=int )
    parser.add_argument( "--metric", 
        help="Metric that will be used to validate the performance of the model. Values must be provided from metric_dispatcher. Ex: 'accuracy'", required=True, type=str )
    
    args = vars(parser.parse_args())
    

    assert args['kind'] in ['classification', 'regression'], f"'kind' should be 'classification' or 'regression'. {args['kind']} was provided"
    
    run(kind=args['kind'], model=args['model'], folds=args['folds'], metric=args['metric'])

