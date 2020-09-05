import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def AUC_KS(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    ks = max(tpr - fpr)
    return [auc, ks]

def onehot_accuracy(ys, pred_ys):
    acc = np.mean(np.argmax(ys, 1) == np.argmax(pred_ys, 1))
    return acc

metric_dict = {
        "auc": roc_auc_score,
        "auc_ks": AUC_KS,
        "acc": onehot_accuracy
    }


def get_metric(metric_name: str):
    metric_name = metric_name.lower()
    return metric_dict[metric_name]

