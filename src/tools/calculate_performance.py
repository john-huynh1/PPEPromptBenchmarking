from sklearn.metrics import confusion_matrix

def create_confusion_matrix(y_true: list, y_pred: list):
    cm = confusion_matrix(y_true, y_pred)
    return cm
def compute_precision(cm: confusion_matrix):
    TN, FP, FN, TP = cm.ravel()
    precision = TP / (TP + FP)
    return precision
def compute_recall(cm: confusion_matrix):
    TN, FP, FN, TP = cm.ravel()
    recall = TN / (TN + FP)
    return recall
def compute_f1(precision:float, recall:float):
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score
