from math import log10, floor
import math
# from pyliftover import LiftOver
import csv


def round_sig(x, sig=6, small_value=1.0e-9):
    if math.isnan(x):
        return x
    return round(x, sig - int(floor(log10(max(abs(x), abs(small_value))))) - 1)

import matplotlib.pyplot as plt

from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

def plot_prediction_recall(y_true, y_scores, label_name = ""):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    shet_auc = auc(recall, precision)
    # summarize scores
    print('auc=%.3f' % (shet_auc))
    # plot the precision-recall curves
    no_skill = len(y_true[y_true==1]) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, marker='.', label=label_name)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    
import pandas as pd

def apply_and_concat(dataframe, field, func, column_names):
    return pd.concat((
        dataframe,
        dataframe[field].apply(
            lambda cell: pd.Series(func(cell), index=column_names))), axis=1)