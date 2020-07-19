# train_model.py
import argparse 
import os
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import gc
import pickle
from datetime import datetime
from time import time
import config
import model_dispatcher
import metric_dispatcher

# load datasets
def load_data(kind):
    
    if kind == 'classification':
        # read the training data
        X_train =  pickle.load(open( f'{config.INPUT_PATH}/X_train.pkl', "rb" )) 
        y_train = pickle.load(open( f'{config.INPUT_PATH}/Y_train.pkl', "rb" ))

        # read validation data
        X_val = pickle.load(open( f'{config.INPUT_PATH}/X_val.pkl', "rb" )) 
        y_val = pickle.load(open( f'{config.INPUT_PATH}/Y_val.pkl', "rb" )) 

        # read test data
        X_test = pickle.load(open( f'{config.INPUT_PATH}/X_test.pkl', "rb" )) 
        y_test = pickle.load(open( f'{config.INPUT_PATH}/Y_test.pkl', "rb" )) 

        return X_train, y_train, X_test, y_test

    else :
        # kind equal to regression
        
        # read the training data
        X_train =  pickle.load(open( f'{config.INPUT_PATH}/X_train.pkl', "rb" )) 
        y_train = pickle.load(open( f'{config.INPUT_PATH}/Y_train.pkl', "rb" ))

        # read validation data
        X_val = pickle.load(open( f'{config.INPUT_PATH}/X_val.pkl', "rb" )) 
        y_val = pickle.load(open( f'{config.INPUT_PATH}/Y_val.pkl', "rb" )) 


        # read test data
        X_test = pickle.load(open( f'{config.INPUT_PATH}/X_test.pkl', "rb" )) 
        y_test = pickle.load(open( f'{config.INPUT_PATH}/Y_test.pkl', "rb" )) 

        return X_train, y_train, X_val, y_val, X_test, y_test



def cv_classification(model, metric, folds, X, y, plot_roc=False):

    cvscores = []
    cv = StratifiedKFold(n_splits=folds, random_state=config.SEED)

    metricfun = metric_dispatcher.metrics_score[metric]

    if (plot_roc & metric=='roc_auc'):

        plot_roc_curve_cv(model, X, y, cv)
    
    else:
        cvscores = cross_val_score(model, X, y, scoring=metricfun, cv=cv, n_jobs=config.NJOBS, error_score='raise', verbose=config.VERBOSE)
    
    return cvscores

    pass


def test_classification(model, metric, X_train, y_train, X_test, y_test):


    # fit the model on training data
    model.fit(X_train, y_train)

    metric_function = metric_dispatcher[metric]

    y_preds = []

    # create predictions for validation samples
    if metric in ['roc_auc']:
        y_preds = model.predict_proba(X_test)
    else:
        y_preds = model.predict(X_test)


    return metric_function(y_test, y_preds)
    pass


def run_regression(model, X_train_vector, Y_train_vector, X_val_vector, Y_val_vector):

    models = []

    FOLDS = len(X_train_vector)


    for fold in range(FOLDS):
        print(f"\n----- Fold: ({fold + 1} / {FOLDS}) -----\n")
        X_trn, X_val = X_train_vector[fold], X_val_vector[fold]
        y_trn, y_val = y_train_vector[fold], y_val_vector[fold]
        

        train_set = lgb.Dataset(
            X_trn,
            label=y_trn,
            categorical_feature=categorical_feature,
        )
        val_set = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=categorical_feature,
        )

        model = lgb.train(
            bst_params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            **fit_params,
        )
        models.append(model)

        del X_trn, X_val, y_trn, y_val
        gc.collect()

    return models

    pass



def run(kind, model, folds, metric):

   

    if kind == 'classification':   
        
        # load train and validation datasets
        X_train, y_train, X_test, y_test = load_data(kind)   

        # fetch the model from model_dispatcher
        clf = model_dispatcher.models[model]
        

        cv_scores = cv_classification(model=clf, metric=metric, folds=folds, X_train, y_train, plot_roc=config.PLOT) 

        print(f'{metric}: %.3f (%.3f)' % (mean(cv_scores), std(cv_scores)))
       

        pass
    else:
        # kind equal to regression

        pass





    # read the training data with folds df = pd.read_csv(config.TRAINING_FILE)
    # training data is where kfold is not equal to provided fold # also, note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True) # drop the label column from dataframe and convert it to
    # a numpy array by using .values.

    # target is label column in the dataframe
    x_train = df_train.drop("label", axis=1).values y_train = df_train.label.values
    # similarly, for validation, we have
    x_valid = df_valid.drop("label", axis=1).values y_valid = df_valid.label.values
    # fetch the model from model_dispatcher
    clf = model_dispatcher.models[model]
    # fir the model on training data
    clf.fit(x_train, y_train)
    # create predictions for validation samples
    preds = clf.predict(x_valid)
    # calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds) print(f"Fold={fold}, Accuracy={accuracy}")
    # save the model
    joblib.dump( clf,
    os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin") )
	




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--kind",   type=str )
    parser.add_argument( "--model",	 type=str )
    parser.add_argument( "--folds",  type=int )
    parser.add_argument( "--metric", type=str )
    args = parser.parse_args()

    assert args.kind not in ['classification', 'regression'], "'kind' should be 'classification' or 'regression'."

    run( fold=args.kind, model=args.model, folds=args.folds, metric=args.metric)