import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, recall_score, precision_score

def cls_metrics(y_true, y_pred):
    try:
        accuracy = accuracy_score(y_true, y_pred)
    except:
        accuracy = np.NaN
    try:
        mf1Score = f1_score(y_true, y_pred, average='macro')
    except:
        mf1Score = np.NaN
    try:
        f1Score  = f1_score(y_true, y_pred, labels = np.unique(y_pred))
    except:
        f1Score = np.NaN
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        area_under_c = auc(fpr, tpr)
    except:
        fpr, tpr = np.NaN, np.NaN
        area_under_c = np.NaN
    try:
        recallScore = recall_score(y_true, y_pred, labels = np.unique(y_pred))
    except:
        recallScore = np.NaN
    try:
        precisionScore = precision_score(y_true, y_pred, labels = np.unique(y_pred))
    except:
        precisionScore = np.NaN
    return dict({"accuracy": accuracy, 'mF1Score': mf1Score, 'f1Score': f1Score, 'auc': area_under_c,
           'precision': precisionScore, 'recall': recallScore})

def convert_iou(y_true, y_pred, threshold=0.5):
    # given y_pred [0,0,0,0,1,1,1,1,0,0], construct a list containing consecutive 1s and 0s as [dict(start=0, end=4, label=0), dict(start=4, end=8, label=1), dict(start=8, end=10, label=0)]

    def get_consecutive(y):
        consecutive = []
        start = 0
        label = y[0]
        for i in range(1, len(y)):
            if y[i] != label:
                consecutive.append(dict(start=start, end=i, label=label))
                start = i
                label = y[i]
        consecutive.append(dict(start=start, end=len(y), label=label))
        return consecutive
    
    iou_true = []
    iou_pred = []
    y_pred_consecutive = get_consecutive(y_pred)
    for y in y_pred_consecutive:
        cur_y_pred = np.zeros(y['end'] - y['start']) + y['label']
        cur_y_true = y_true[y['start']:y['end']]
        intersection = np.sum(cur_y_pred * cur_y_true)
        union = np.sum(cur_y_pred) + np.sum(cur_y_true) - intersection
        if union == 0:
            iou = 1
        else:
            iou = intersection / union
        if iou >= threshold:
            iou_pred.append(y['label'])
            iou_true.append(y['label'])
        else:
            iou_pred.append(y['label'])
            iou_true.append(1-y['label'])
    
    iou_true = np.array(iou_true)
    iou_pred = np.array(iou_pred)

    return iou_true, iou_pred
