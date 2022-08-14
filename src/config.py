# config.py

class Config:
    SEED = 42
    SEC_PER_MIN = 60
    EPS = 1e-15
    INPUT_PATH = "../data/"
    MODEL_OUTPUT = "../models/"
    SAVE_MODEL = True
    TEST_PATH = "data/features/"
    SUBMIT_PATH = "data/submission/"
    KAGGLE_PROJECT = "interbank20"
    KAGGLE_SUBMIT = True
    NJOBS = -1
    USEGPU = False
    VERBOSE = 150
    PLOT = False
    SEED = 47
    ESR = 100            # early_stopping_rounds 
    LABEL = "target"
    FOLDS = 5
    KEY_ID = "customer_ID"
    TASK = "classification"
