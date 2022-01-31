# data_functions.py
#!/usr/bin/env ml
# coding: utf-8
# Import libraries
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import gc
from scipy import stats


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
def load_data(file_path, kind='csv'):
    data = pd.DataFrame([])

    if kind=='csv':
        data  = pd.read_csv(f"{file_path}.csv", sep=config.CSV_SEP).pipe(reduce_mem_usage)
    elif kind=='pickle':
        data  = pd.read_pickle(f"{file_path}.pkl").pipe(reduce_mem_usage)
    elif kind=='parquet':
        data  = pd.read_parquet(f"{file_path}.parquet").pipe(reduce_mem_usage)
    else:
        raise Exception(f"`kind` should be csv, pickle or parquet. `{kind}` value is not allowed.") 
    return data


# SAVE DATASET
def save_data(output_file_name, data):
    
    data.to_pickle(config.OUTPUT_PATH + f'{output_file_name}.pkl')
    
    return 0


# replace outliers with top and bottom value
# @data: imput dataset
# @attrs: what variables want to trim their outlier's values
# @return: same dataset from imput without outliers 
def winsorizer(data, attrs, params=None):
    for x in attrs:
        q75,q25 = np.percentile(data.loc[:,x],[75,25])
        intr_qr = q75-q25
    
        max = q75+(1.5*intr_qr)
        min = q25-(1.5*intr_qr)
    
        data.loc[data[x] < min,x] = min
        data.loc[data[x] > max,x] = max
    
    return data

# apply power transformation to numerical variables in order to stabilize variance
# @data: imput dataset
# @attrs: what variables want to apply power transformation
# @funtion: function to apply. Only functions allowed are 'ln', 'sqrt', 'pow' and 'boxcox'
# @return: same dataset from imput with a new numerical values distribution
def power_transformation(data, attrs, function='ln'):

    assert function in ['ln', 'sqrt', 'pow', 'boxcox'], "function must be 'ln', 'sqrt', 'pow' or 'boxcox'"

    if function == 'ln':
        data.loc[:, attrs] =  data.loc[:, attrs].apply(lambda x: np.log1p(x))

    elif function == 'sqrt':
        data.loc[:, attrs] = data.loc[:, attrs].apply(lambda x: np.sqrt(abs(x)))

    elif function == 'pow':
        data.loc[:, attrs] = data.loc[:, attrs].apply(lambda x: np.power(x,2))

    elif function == 'boxcox':
        data.loc[:, attrs] = data.loc[:, attrs].apply(lambda x: stats.boxcox(x))
    else:
        data.loc[:, attrs] =  data.loc[:, attrs].apply(lambda x: np.log1p(x))
    
    return data 