from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from io import open
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from scipy import sparse

import csv
import datetime
import heapq
import json
import os
import pickle
import time

import h5py
import pickle
import numpy as np
import pandas as pd
import pyarrow.feather as feather



def is_number(s):
    """Check if a string is a number or not."""

    try:
        float(s)
        return True
    except ValueError:
        return False


def save_csv(path, df):
    """Save data as a CSV file.

    Args:
        X (pandas dataframes): Data matrix
        path (str): Path to the CSV file to save data.
    """

    df.to_csv(path, index=False)


def save_libsvm(X, y, path):
    """Save data as a LibSVM file.

    Args:
        X (numpy or scipy sparse matrix): Data matrix
        y (numpy array): Target vector.
        path (str): Path to the CSV file to save data.
    """

    dump_svmlight_file(X, y, path, zero_based=False)


def save_hdf5(X, y, path):
    """Save data as a HDF5 file.

    Args:
        X (numpy or scipy sparse matrix): Data matrix
        y (numpy array): Target vector.
        path (str): Path to the HDF5 file to save data.
    """

    with h5py.File(path, 'w') as f:
        is_sparse = 1 if sparse.issparse(X) else 0
        f['issparse'] = is_sparse
        f['target'] = y

        if is_sparse:
            if not sparse.isspmatrix_csr(X):
                X = X.tocsr()

            f['shape'] = np.array(X.shape)
            f['data'] = X.data
            f['indices'] = X.indices
            f['indptr'] = X.indptr
        else:
            f['data'] = X


def save_pickle(path, df):
    """Save data as a Pickle file.

    Args:
        X (pandas DataFrame): Dataframe
        
        path (str): Path to the Pickle file to save data.
    """

    # open a file, where you ant to store the data
    with open(path, 'ab') as f:
      
        # source, destination
        pickle.dump(df, f)                     


def save_feather(path, df):
    """Save data as a feather file.

    Args:
        X (pandas DataFrame): Dataframe
        
        path (str): Path to the feather file to save data.
    """

    feather.write_feather(df, path, compression='lz4')



def save_data(path, df):
    """Save data as a CSV, LibSVM or HDF5 file based on the file extension.

    Args:
        df (pandas DataFrame): Dataframe
        path (str): Path to the CSV, LibSVM or HDF5 file to save data.
    """
    catalog = {'.csv': save_csv, '.sps': save_libsvm, '.h5': save_hdf5, 
                '.pkl': save_pickle, '.ftr': save_feather,}

    ext = os.path.splitext(path)[1]
    
    assert ext in catalog.keys(), f"'Format file `{ext}` is not supported by function load_data."

    func = catalog[ext]
    
    func(path, df)

    pass



def load_data(path):
    """Load data from a CSV, LibSVM or HDF5 file based on the file extension.

    Args:
        path (str): A path to the CSV, LibSVM or HDF5 format file.
        

    Returns:
        pandas DataFrame
    """

    catalog = {'.csv': load_csv, '.sps': load_svmlight_file, 
                '.h5': load_hdf5, '.ftr': load_feather, '.pkl':load_pickle,
                'xls': load_excel}

    ext = os.path.splitext(path)[1]

    assert ext in catalog.keys(), f"'Format file `{ext}` is not supported by function load_data."

    func = catalog[ext]
    
    return func(path)



def load_csv(path):
    """Load data from a CSV file.

    Args:
        path (str): A path to the CSV format file containing data.
        
    Returns:
        Pandas Dataframe 
    """

    return pd.read_csv(path)



def load_excel(path):
    """
    Load data from a xls file.

    Args:
        path (str): A path to the xls format file containing data.
        
    Returns:
        Pandas Dataframe 
    """
    return pd.read_excel(path)


def load_feather(path):
    """Load data from a feather file (https://arrow.apache.org/docs/python/feather.html).

    Args:
        path (str): A path to the feather file containing data.
         
        
    Returns:
        Pandas Dataframe 
    """

    return pd.read_feather(path)


def load_hdf5(path):
    """Load data from a HDF5 file.

    Args:
        path (str): A path to the HDF5 format file containing data.
        dense (boolean): An optional variable indicating if the return matrix
                         should be dense.  By default, it is false.

    Returns:
        Data matrix X and target vector y
    """

    with h5py.File(path, 'r') as f:
        is_sparse = f['issparse'][...]
        if is_sparse:
            shape = tuple(f['shape'][...])
            data = f['data'][...]
            indices = f['indices'][...]
            indptr = f['indptr'][...]
            X = sparse.csr_matrix((data, indices, indptr), shape=shape)
        else:
            X = f['data'][...]

        y = f['target'][...]

    return X, y



def load_pickle(path):
    """Load data from a Pickle file.

    Args:
        path (str): A path to the CSV format file containing data.
        

    Returns:
        Data matrix X and target vector y
    """

    with open(path, 'rb') as f:
        X = pickle.load(file)

    y = np.array(X[:, 0]).flatten()
    X = X[:, 1:]

    return X, y



def read_sps(path):
    """Read a LibSVM file line-by-line.

    Args:
        path (str): A path to the LibSVM file to read.

    Yields:
        data (list) and target (int).
    """

    for line in open(path):
        # parse x
        xs = line.rstrip().split(' ')

        yield xs[1:], int(xs[0])



def shuf_file(f, shuf_win):
    heap = []
    for line in f:
        key = hash(line)
        if len(heap) < shuf_win:
            heapq.heappush(heap, (key, line))
        else:
            _, out = heapq.heappushpop(heap, (key, line))
            yield out

    while len(heap) > 0:
        _, out = heapq.heappop(heap)
        yield out



class PathJoiner:
    """Load directory names from SETTINGS.json.

    Originally written by Baris Umog (https://www.kaggle.com/barisumog).

    Usage:
        # In SETTINGS.json, "data": "/path/to/data/".
        # To load "/path/to/data/targets.array" file to y:
        PATH = PathJoiner()
        y = load(PATH.data('targets.array'))
    """

    def __init__(self, filename='SETTINGS.json'):
        with open(filename) as file:
            self.subdirs = json.load(file)

    def __getattr__(self, attr):
        subdir = self.subdirs[attr]
        return lambda *dirs: os.path.join(subdir, *dirs)


def stream_lines(filename, encoding='utf-8', ignore_errors=False):
    errors = 'ignore' if ignore_errors else 'strict'
    with open(filename, encoding=encoding, errors=errors) as file:
        for line in file:
            yield line


def stream_csv(filename, encoding='utf-8', ignore_errors=False):
    stream = stream_lines(filename, encoding, ignore_errors)
    return csv.reader(stream)


def limit_stream(stream, count=1, skip=0):
    for i in range(skip):
        next(stream)
    for i in range(count):
        yield next(stream)


def save_obj(filename, obj):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info('saved : {}\t{}'.format(filename, type(obj)))


def load_obj(filename):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    logger.info('loaded : {}\t{}'.format(filename, type(obj)))
    return obj


def save_array(filename, X):
    with h5py.File(filename, 'w') as file:
        file['data'] = X
    logger.info('saved : {}\t{}\t{}'.format(filename, X.dtype, X.shape))


def load_array(filename):
    with h5py.File(filename, 'r') as file:
        X = file['data'][...]
    logger.info('loaded : {}\t{}\t{}'.format(filename, X.dtype, X.shape))
    return X


def save_sparse(filename, X):
    with h5py.File(filename, 'w') as file:
        file['shape'] = np.array(X.shape)
        file['data'] = X.data
        file['indices'] = X.indices
        file['indptr'] = X.indptr
    logger.info('saved : {}\t{}\t{}'.format(filename, X.dtype, X.shape))


def load_sparse(filename):
    with h5py.File(filename, 'r') as file:
        shape = tuple(file['shape'][...])
        data = file['data'][...]
        indices = file['indices'][...]
        indptr = file['indptr'][...]
    X = sparse.csr_matrix((data, indices, indptr), shape=shape)
    logger.info('loaded : {}\t{}\t{}'.format(filename, X.dtype, X.shape))
    return X


def save(filename, X):
    catalog = {'obj': save_obj, 'array': save_array, 'sparse': save_sparse}
    extension = filename.split('.')[-1]
    func = catalog[extension]
    func(filename, X)


def load(filename):
    catalog = {'obj': load_obj, 'array': load_array, 'sparse': load_sparse}
    extension = filename.split('.')[-1]
    func = catalog[extension]
    X = func(filename)
    return X


class Clock(object):

    def __init__(self):
        self.start = time.time()
        self.last = self.start
        self.now = self.start
        self.report()

    def check(self):
        self.now = time.time()
        self.report()
        self.last = self.now

    def report(self):
        txt = '\n[CLOCK]  [ {} ]    '
        txt += 'since start: [ {} ]    since last: [ {} ]\n'
        current = time.asctime().split()[3]
        since_start = datetime.timedelta(seconds=round(self.now - self.start))
        since_last = datetime.timedelta(seconds=round(self.now - self.last))
        logger.info(txt.format(current, since_start, since_last))


def beep(n=1):
    for _ in range(n):
        os.system('beep')


def print_shape_type(*objs):
    for obj in objs:
        try:
            logger.info(obj.shape, obj.dtype, type(obj))
        except AttributeError:
            logger.error(obj.shape, type(obj))