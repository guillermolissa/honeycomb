# config.py
INPUT_PATH = "data/raw/"
OUTPUT_PATH = "data/train_test/"
TASK = "classification"
NJOBS = -1
VERBOSE = 1
PLOT = False
SEED = 47
TEST_SIZE = 0.3


# Specify numerical and categorical features in order to be processed adecually
num_features = []

cat_features = []

# Specify which would be the target feature
LABEL = "target"