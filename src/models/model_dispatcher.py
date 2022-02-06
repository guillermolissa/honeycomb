# model_dispatcher.py
from sklearn import ensemble 
from sklearn import tree
from sklearn import naive_bayes
from sklearn import linear_model

models = {
"decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini" ),

"decision_tree_entropy": tree.DecisionTreeClassifier( criterion="entropy"),

"rf": ensemble.RandomForestClassifier(),

"naive_bayes": naive_bayes.MultinomialNB(),

"logreg": linear_model.LogisticRegression()

}