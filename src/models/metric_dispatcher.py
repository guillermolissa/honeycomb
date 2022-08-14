from metrics import mcc, apk
from sklearn.metrics import roc_auc_score, accuracy_score


metrics_score = {
"roc_auc": roc_auc_score,
"accuracy": accuracy_score,
"mcc": mcc,
"apk": apk
}

