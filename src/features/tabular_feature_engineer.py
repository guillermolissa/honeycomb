# functions.py
#!/usr/bin/env python
# coding: utf-8
# Import libraries
# many functions useful for scripts in features folder
import pandas as pd
import numpy as np
from sklearn import preprocessing
import category_encoders as encoders
from pathlib import Path
import argparse
import pickle
import config
import gc
import os
from google.cloud import storage
from joblib import Parallel, delayed
import multiprocessing

# REDUCE MEMORY USAGE
def reduce_mem_usage(df, verbose=False):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    int_columns = df.select_dtypes(include=["int"]).columns
    float_columns = df.select_dtypes(include=["float"]).columns

    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

# LOAD DATASET 
def load_data(input_file, kind='csv'):
    data = pd.DataFrame([])

    if kind=='csv':
        data  = pd.read_csv(f"{config.INPUT_PATH}{input_file}.csv", sep=config.CSV_SEP).pipe(reduce_mem_usage)
    elif kind=='pickle':
        data  = pd.read_pickle(f"{config.INPUT_PATH}{input_file}.pkl").pipe(reduce_mem_usage)
    elif kind=='parquet':
        data  = pd.read_parquet(f"{config.INPUT_PATH}{input_file}.parquet").pipe(reduce_mem_usage)
    else:
        raise Exception(f"`kind` should be csv, pickle or parquet. `{kind}` value is not allowed.") 
    return data

# SAVE DATASET
def save_data(output_file, data):
    data.to_pickle(f"{config.OUTPUT_PATH}{output_file}.pkl")

    if config.USE_GCP:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(config.GCP_BUCKET_NAME)

        blob = bucket.blob(f"{config.GCP_BUCKET_FOLDER_NAME}{output_file}.pkl")
        blob.upload_from_filename(f"{config.OUTPUT_PATH}{output_file}.pkl")

        # remove local file because it is now in the bucket
        os.remove(f"{config.OUTPUT_PATH}{output_file}.pkl")
    
    return 0

# =====================  FUNCTIONS TO BUILD COMMON FEATURES  ==================== #


# GET SIMPLE STATISTICS VALUES
def get_simple_stats(data, attrs, prefix_sep='_stats_',params=None):
    statfunc = []
    for c in attrs:
        dfstat = data.loc[:,attrs].agg({c:statfunc})
    dfstat.columns = [f"{c}_{v}" for v in statfunc]
    dfstat.reset_index(inplace=True)
    
    return dfstat







# combine categorical data from attrs list and create a new categorical variable from them
# esto deberia estar en data folder
def combine_cat(data, attrs, prefix_sep='_'):
    new_attr = prefix_sep.join(attrs)
    
    val_vector = []

    for attr in attrs:
        val_vector.append(data[attr].astype(str).values + prefix_sep)

    new_values = val_vector[0]
    for idx in range(1,len(val_vector)):
        new_values += val_vector[1]


    data[new_attr] = new_values
    return data



# Create dummies variables from categorical variables
# @data: dataframe input (only pandas DataFrames are allowed)
# @attrs: wich variable would you like to convert to dummies
# @return: The outputs are a new dataset with new attributes and a list of these attributes.
def make_cat_dummy(data, attrs=None, prefix_sep='_ohe_', params=None):
    _data = pd.get_dummies(data.loc[:,attrs], prefix_sep=prefix_sep, dummy_na=True)
    new_attrs = _data.columns.tolist()

    return _data, new_attrs


# Create Label encode variables from categorical variables
# @data: dataframe input (only pandas DataFrames are allowed)
# @attrs: wich variable would you like to convert to dummies
# @return: The outputs are a new dataset with new attributes and a list of these attributes.
def make_cat_le(data, attrs, prefix_sep='le_', params=None):
    
    # since its categorical data, we fillna with a string
    # and we convert all the data to string type
    # so, no matter its int or float, its converted to string # int/float but categorical!!!
    _data = data.loc[:,attrs].fillna('unknown').astype(str)
    
    # keep labelEncode object in a dict for each variable
    le_struct = dict()
    for c in attrs:
        le = preprocessing.LabelEncoder()
        le = le.fit(_data.loc[:,c].values)
        _data.loc[:,c] = le.transform(_data.loc[:,c].values)
        le_struct[c] = le

    # rename columns name with prefix
    _data.columns = [prefix_sep + c for c in _data.columns]
    new_attrs = _data.columns.tolist()

    return _data, new_attrs, le_struct


# Create Label encode variables from categorical variables
# @data: dataframe input (only pandas DataFrames are allowed)
# @attrs: wich variable would you like to convert to dummies
# @return: The outputs are a new dataset with new attributes and a list of these attributes.
def make_cat_from_le(data, le_dict, prefix_sep='le_'):
    
    # since its categorical data, we fillna with a string
    # and we convert all the data to string type
    # so, no matter its int or float, its converted to string # int/float but categorical!!!
    attrs = list(le_dict.keys())
    _data = data.loc[:,attrs].fillna('unknown').astype(str)
    

    for c in attrs:
        le = le_dict[c]
        _data.loc[:,c] = le.transform(_data.loc[:,c].values)

    # rename columns name with prefix
    _data.columns = [prefix_sep + c for c in _data.columns]
    new_attrs = _data.columns.tolist()

    return _data, new_attrs


def make_cat_woe(X, y, attrs, prefix_sep='woe_', params=None):

    # since its categorical data, we fillna with a string
    # and we convert all the data to string type
    # so, no matter its int or float, its converted to string # int/float but categorical!!!
    X = X.loc[:,attrs].astype(str)

    woe = encoders.WOEEncoder(cols=attrs, return_df=True, handle_missing='value')

    # keep WOEEncoder object in order to be used in the Test set
    woe = woe.fit(X, y)
    _data = woe.transform(X)

    # rename columns name with prefix
    _data.columns = [prefix_sep + c for c in _data.columns]
    new_attrs = _data.columns.tolist()

    return _data, new_attrs, woe


def apply_parallel(data_grouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in data_grouped)
    return pd.concat(retLst)


def make_numeric_feateng(df):
    feateng_vector = []

    for p in [1, 3, 6]:
        
        df_tmp =df.apply(
            lambda x: x.diff(p)
        )
        
        df_tmp.columns = [c + f"_diff_p{p}" for c in df_tmp.columns]
        
        feateng_vector.append(df_tmp)


    # Moving sum 
    for window in [3, 6, 9, 12]:
        
        df_tmp = df.apply(
            lambda x: x.rolling(window).sum(skipna=True)
        )
        df_tmp.columns = [c + f"_diff_p{window}" for c in df_tmp.columns]
        feateng_vector.append(df_tmp)
    

    # Moving average
    for window in [3, 6, 9, 12]:

        df_tmp = df.apply(
            lambda x: x.rolling(window, min_periods = 1).mean(skipna=True)
        )
        df_tmp.columns = [c + f"_avg_p{window}" for c in df_tmp.columns]
        feateng_vector.append(df_tmp)


    # Rolling Max 
    for window in [3, 6, 9, 12]:

        df_tmp = df.apply(
            lambda x: x.rolling(window, min_periods = 1).max(skipna=True)
        )
        df_tmp.columns = [c + f"_max_p{window}" for c in df_tmp.columns]
        feateng_vector.append(df_tmp) 


    # Rolling Min 
    for window in [3, 6, 9, 12]:

        df_tmp = df.apply(
            lambda x: x.rolling(window, min_periods = 1).min(skipna=True)
        )
        df_tmp.columns = [c + f"_min_p{window}" for c in df_tmp.columns]
        feateng_vector.append(df_tmp) 


    # Rolling Std 
    for window in [3, 6, 9, 12]:

        df_tmp = df.apply(
            lambda x: x.rolling(window, min_periods = 1).std(skipna=True)
        )
        df_tmp.columns = [c + f"_std_p{window}" for c in df_tmp.columns]
        feateng_vector.append(df_tmp) 


    # Percentage change between the current and a prior element.
    for p in [1, 3, 6]:

        df_tmp = df.apply(
            lambda x: x.pct_change(periods=p, fill_method='pad').replace(np.inf,0).replace(-np.inf,0).replace(np.nan,0)
        )
        df_tmp.columns = [c + f"_pct_p{p}" for c in df_tmp.columns]
        feateng_vector.append(df_tmp) 


    return pd.concat(feateng_vector, axis=1)