"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""

import numpy as np


def confusion_matrix(y_true, y_pred):
    """
    returns confusion matrix
    [[TN, FP],
     [FN, TP]]
    :param y_true:
    :param y_pred:
    :return:
    """
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        else:
            FN += 1
    return np.array([[TN, FP], [FN, TP]])


def accuracy(y_true, y_pred):
    """
    accuracy = (TN + TP) / (TN + TP + FP + FN)
    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    cm = confusion_matrix(y_true, y_pred)
    return (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])


def precision_score(y_true, y_pred):
    """
    precision = TP / (TP + FP)
    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    cm = confusion_matrix(y_true, y_pred)
    return cm[1][1] / (cm[1][1] + cm[0][1])


def recall_score(y_true, y_pred):
    """
    recall = TP / (TP + FN)
    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    cm = confusion_matrix(y_true, y_pred)
    return cm[1][1] / (cm[1][1] + cm[1][0])


def f1_score(y_true, y_pred):
    """
    f1 = 2 * (precision * recall) / (precision + recall)
    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)
