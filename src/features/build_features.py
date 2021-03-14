# make_dataset.py
#!/usr/bin/env python
# coding: utf-8
# Import libraries
import logging
import pandas as pd
from pathlib import Path
import argparse
import pickle
import config
from feateng_functions import load_data, save_data
from feateng_functions import build_numerical_feateng, build_cat_le, build_cat_from_le



def main(input_file, output_file):
    """ 
        You must define here how to build any features engineer from data
        :data: dataset from where features will be made
        :features: final dataset with all features eng
    """

    logger = logging.getLogger(__name__)
    logger.info('INIT: build features engineer from processed data')

    logger.info('RUN: loading data')
    df = load_data(input_file, kind='csv')
   
    logger.info(f'RUN: data size before be processed: {df.shape}')
    
    logger.info(f'RUN: building features')

   
    # ****************************************************** # 
    # put here what you think is needed to build features 
    # ****************************************************** #
    
    features = None
    del df 

    logger.info(f'RUN: features size : {features.shape}')

    logger.info(f'RUN: saving features')
    save_data(output_file, features)

    logger.info('END: making features data set has finished.')

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]
    
    parser = argparse.ArgumentParser()
    parser.add_argument( "--input_file",   required=True, 
        help="Specify file name without extension from processed folder from where features will be built.", type=str)
    parser.add_argument( "--output_file",   required=True, 
        help="Specify output file name which will be saved into features folder.", type=str)

    args = vars(parser.parse_args())
    
    main(input_file=args['input_file'], output_file=args['output_file'])