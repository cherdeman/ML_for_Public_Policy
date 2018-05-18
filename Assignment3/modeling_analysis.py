from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

import data_processing

##################################################
############## Primary Functions #################
##################################################

def master_loop_with_time(df, start_time_date, end_time_date, prediction_windows, feature_cols, predictor_col, models_to_run, clfs, grid):
    '''
    '''
    #start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    #end_time_date = datetime.strptime(end_time, '%Y-%m-%d')
    test_output = []
    for prediction_window in prediction_windows:
        train_start_time = start_time_date
        train_end_time = train_start_time + relativedelta(months=+prediction_window) - relativedelta(days=+1)
        while train_end_time + relativedelta(months=+prediction_window)<=end_time_date:
            test_start_time = train_end_time + relativedelta(days=+1)
            test_end_time = test_start_time + relativedelta(months=+prediction_window) - relativedelta(days=+1)
            
            print('training date range:', train_start_time, train_end_time) 
            print('testing date range:', test_start_time, test_end_time)
            # Build training and testing sets
            X_train, y_train, X_test, y_test = temporal_train_test_sets(df, train_start_time, train_end_time, test_start_time, test_end_time, feature_cols, predictor_col)
            # Fill nulls here to avoid data leakage
            X_train = data_processing.fill_nulls(X_train,'students_reached')
            X_test = data_processing.fill_nulls(X_test,'students_reached')
            # Build classifiers
            row_lst = clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test, (train_start_time,train_end_time), (test_start_time,test_end_time))
            # Increment time
            train_end_time += relativedelta(months=+prediction_window)
            test_output.extend(row_lst)

    output_df = pd.DataFrame(test_output, columns=('training_dates', 'testing_dates', 'model_type','clf', 
        'parameters', 'baseline', 'auc-roc','a_at_5', 'a_at_20', 'a_at_50', 'f1_at_5', 'f1_at_20', 'f1_at_50', 'p_at_1','p_at_5', 'p_at_10', 'p_at_20','p_at_50','r_at_1','r_at_5', 'r_at_10', 'r_at_20','r_at_50'))

    return output_df


def clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test, training_dates, testing_dates):
    """Runs the loop using models_to_run, clfs, gridm and the data
    """
    results = []
    for n in range(1, 2):
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    model_fit = clf.fit(X_train, y_train)
                    y_pred_probs = model_fit.predict_proba(X_test)[:,1]
                    # Store metrics and model info for comparison
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    row = [training_dates, testing_dates, models_to_run[index],clf, p,
                                                       baseline(y_test),   
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       accuracy_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       accuracy_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       accuracy_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                       f1_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       f1_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       f1_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                       ]
                    results.append(row)
                    plot_precision_recall_n(y_test,y_pred_probs,clf)
                except IndexError as e:
                    print('Error:',e)
                    continue
    
    return results


##################################################
############## Helper Functions ##################
##################################################

# Train Test Split
def temporal_train_test_sets(df, train_start, train_end, test_start, test_end, feature_cols, predictor_col):
    '''
    '''
    train_df = df[(df['date_posted'] >= train_start) & (df['date_posted'] <= train_end)]
    test_df = df[(df['date_posted'] >= test_start) & (df['date_posted'] <= test_end)]

    X_train = train_df[feature_cols]
    y_train = train_df[predictor_col]

    X_test = test_df[feature_cols]
    y_test = test_df[predictor_col]

    return X_train, y_train, X_test, y_test

# ML Validation Metrics
def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    recall = recall_score(y_true, preds_at_k)

    return recall

def baseline(y_test):
    base = y_test.sum()/ len(y_test)

    return base

def accuracy_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    #print(len(preds_at_k))

    pred = accuracy_score(y_true, preds_at_k)

    return pred

def f1_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)

    f1 = f1_score(y_true, preds_at_k)

    return f1

def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    plt.show()
    
