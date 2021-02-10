# hyperparams_dispatcher.py
from sklearn import ensemble 
from sklearn import tree
from xgboost.sklearn import XGBClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



models = {


"lgbm": {
    'n_estimators': [i for i in range(50,1000,50)],
    'learning_rate' : [0.01, 0.2],
    'min_data_in_leaf' : [i for i in range(20,1000,40)],
    'max_depth' : [i for i in range(3,10,2)],
    'gamma': [i/10.0 for i in range(0,5)],
    'subsample': [i/10.0 for i in range(6,10)],
    'colsample_bytree' : [i/10.0 for i in range(6,10)],
    'bagging_freq' : [3, 5, 10, 20, 30],
    'reg_lambda' : [i/10.0 for i in range(4,10)],
    'reg_alpha' : [0, 0.001, 0.005, 0.01, 0.05],
},

"xgb": {
    'n_estimators': [i for i in range(50,1000,50)],
    'learning_rate' : [0.01, 0.2],
    'min_child_weight' : [i for i in range(1,10,2)],
    'max_depth' : [i for i in range(3,10,2)] ,
    'gamma': [i/10.0 for i in range(0,5)],
    'subsample': [i/10.0 for i in range(6,10)],
    'colsample_bytree' : [i/10.0 for i in range(6,10)],
    'reg_lambda' : [i/10.0 for i in range(4,10)],
    'reg_alpha' : [0, 0.001, 0.005, 0.01, 0.05],
}
}