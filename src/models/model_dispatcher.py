# model_dispatcher.py
from sklearn import ensemble 
from sklearn import tree
#from sklearn import naive_bayes
#from sklearn import linear_model
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

models = {
"id3_gini": tree.DecisionTreeClassifier(criterion="gini" ),

"id3_entropy": tree.DecisionTreeClassifier( criterion="entropy"),

"rf": ensemble.RandomForestClassifier(n_estimators=200, max_depth=5, n_jobs=-1),

"lgb": LGBMClassifier(n_estimators=200, max_depth=5,
                          verbose=-1,
                          learning_rate=0.03, reg_lambda=50,
                          min_child_samples=2400,
                          num_leaves=95,
                          max_bins=511, random_state=47,  n_jobs=-1),

"xgb": XGBClassifier(learning_rate =0.1,
 n_estimators=200,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

#"naive_bayes": naive_bayes.MultinomialNB(),

#"logreg": linear_model.LogisticRegression()

}