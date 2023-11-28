from __future__ import division, print_function
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('agg')
from scipy.optimize import linear_sum_assignment as linear_assignment
import random
import os
import argparse

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

from sklearn import metrics
import time

# -------------------------------
# Evaluation Criteria
# -------------------------------
def match_cluster_miou(y_true, y_pred, y_weight, base_c, class_num, over_c=0, fix=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """

    if fix:
        y_pred = y_pred[y_true >= base_c]
        y_weight = y_weight[y_true >= base_c]
        y_true = y_true[y_true >= base_c]
        
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = over_c + class_num
    #max(y_pred.max(), y_true.max()) + 1
    if fix:
        w = np.zeros((D - base_c, D - base_c - over_c), dtype=float)
    else:
        w = np.zeros((D, D - over_c), dtype=float)

    for i in range(y_pred.size):
        if fix and y_pred[i] >= base_c:
            w[y_pred[i] - base_c, y_true[i] - base_c] += y_weight[i]
        elif not fix:
            w[y_pred[i], y_true[i]] += y_weight[i]

    row = w.shape[0]
    new_ind = np.zeros(row, dtype=int)
    base_ind = np.array([i for i in range(base_c)], dtype=int)
    
    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T
    row_ind = ind[:, 0]
    col_ind = ind[:, 1]
    if fix:
        for i, row_label in enumerate(row_ind):
            new_ind[row_label] = col_ind[i] + base_c
        ind = np.concatenate([base_ind, new_ind])
    else:
        for i, row_label in enumerate(row_ind):
            new_ind[row_label] = col_ind[i]
        ind = new_ind
    
    # ind = []
    # while(row > 0):
    #     ind = linear_assignment(w.max() - w)
    #     ind = np.vstack(ind).T

    #     row_ind = ind[:, 0]
    #     col_ind = ind[:, 1]
    #     if fix:
    #         for i, row_label in enumerate(row_ind):
    #             new_ind[row_label] = col_ind[i] + base_c
    #         ind = np.concatenate([base_ind, new_ind])
    #     else:
    #         for i, row_label in enumerate(row_ind):
    #             new_ind[row_label] = col_ind[i]
    #         ind = new_ind
    #     w = w[row_ind, :]
    #     row -= row_ind.shape[0]  

    return ind