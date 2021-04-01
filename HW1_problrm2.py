import numpy as np
import pandas as pd

file = np.loadtxt('HW2_labels.txt',  delimiter=',')
y_predict, y_true = file[:, :2], file[:, -1]

def tp_fp_fn_tn(y_true, y_predict, percent=None):
    if percent == None:
        percent = 50

    else:
        top = int(np.shape(y_predict)[0] * percent // 100)
        y_predict = y_predict[:top]
        y_true = y_true[:top]

    y_binar_predict = [np.shape(y_true)]
    y_binar_predict = ((y_predict[:,1]) >= percent / 100) * 1

    TP = np.sum((y_true == 1) & (y_binar_predict == 1))
    FP = np.sum((y_true == 0) & (y_binar_predict == 1))
    FN = np.sum((y_true == 1) & (y_binar_predict == 0))
    TN = np.sum((y_true == 0) & (y_binar_predict == 0))

    return [TP, FP, FN, TN]

def precision_score(y_true, y_predict, percent=None):
    TP = tp_fp_fn_tn(y_true, y_predict, percent=None)[0]
    FP = tp_fp_fn_tn(y_true, y_predict, percent=None)[1]
    return TP/(TP+FP)

def recall_score(y_true, y_predict, percent=None):
    TP = tp_fp_fn_tn(y_true, y_predict, percent=None)[0]
    FN = tp_fp_fn_tn(y_true, y_predict, percent=None)[2]
    return TP/(TP+FN)

def accuracy_score(y_true, y_predict, percent=None):
    TP = tp_fp_fn_tn(y_true, y_predict, percent=None)[0]
    FP = tp_fp_fn_tn(y_true, y_predict, percent=None)[1]
    FN = tp_fp_fn_tn(y_true, y_predict, percent=None)[2]
    TN = tp_fp_fn_tn(y_true, y_predict, percent=None)[3]
    return (TP+TN)/(TP+FP+FN+TN)

def lift_score(y_true, y_predict, percent=None):
    TP = tp_fp_fn_tn(y_true, y_predict, percent=None)[0]
    FP = tp_fp_fn_tn(y_true, y_predict, percent=None)[1]
    FN = tp_fp_fn_tn(y_true, y_predict, percent=None)[2]

    l = np.shape(y_true)[0]
    return  TP / (TP+FP) / (TP+FN) / l

def f1_score(y_true, y_predict, percent=None):
    return 2 * precision_score(y_true, y_predict, percent=None) * recall_score(y_true, y_predict, percent=None) / (precision_score(y_true, y_predict, percent=None) + recall_score(y_true, y_predict, percent=None))
