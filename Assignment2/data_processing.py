import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.metrics as metrics
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

path = 'data/credit-data.csv'

#### LOAD DATA ####
def load_data(path, index_col = None):
	'''
	'''
	if os.path.exists(path):
	    df = pd.read_csv(path)
	else:
		raise Exception('The file does not exist at this location')

	return df


#### EXPLORE DATA ####
def make_histogram(df, var_of_interest, kde = True, rug = False):
	'''
	Used for continuous/numerical variables
	'''
	# Note: NaN's are exluded to get a sense of the distribution before cleaning
	plot_var = df[var_of_interest]
	sns.distplot(plot_var[~plot_var.isnull()], kde = kde, rug = rug)
	plt.title(var_of_interest + ' Histogram')
	plt.ylabel('Frequency')
	plt.show()


def make_countchart(df, var_of_interest):
	'''
	'''
	plot_var = df[var_of_interest]
	sns.countplot(plot_var, data=df)
	plt.title(var_of_interest + ' Countplot')
	plt.show()


def check_correlations(df):
	'''
	'''
	return df.corr()

def find_high_corr(corr_matrix, threshold, predictor_var):
	'''
	Find all variables that are highly correlated with the predictor and thus 
	likely candidates to exclude
	''' 
	return corr_matrix[corr_matrix[predictor_var] > threshold].index

def plot_correlations(df, x, y, hue = None, fitreg = False):
	'''
	'''
	sns.lmplot(x, y, hue = hue, fit_reg = fitreg, data = df)
	plt.title(x + ' vs ' + y)
	plt.show()


#### DATA PRE-PROCESSING/CLEANING ####
def fill_nulls(df):
	'''
	'''
	# Find columns with missing values
	isnull = df.isnull().any()
	isnull_cols = list(isnull[isnull == True].index)

	# Fill nulls with median, the distributions are characterised by long tails
	# that will pull the mean up, making the median a better estimate of a 
	# likely value
	for col in isnull_cols:
		col_mean = df[col].median()
		df[col].fillna(col_mean, inplace = True)

	return df


#### GENERATE FEATURES/PREDICTIONS ####
def x_y_generator(df, feature_cols, predictor_col):
	'''
	'''
	all_cols = df.columns
	drop_cols = set(all_cols) - set(feature_cols)

	X = df.drop(drop_cols, axis = 1)

	Y = df[predictor_col]

	return X, Y

def cat_to_dummy(df, var_of_interest):
	'''
	'''
	return pd.get_dummies(df, columns = [var_of_interest])

def continuous_to_cat(df, var_of_interest, bins = 10, labels = False):
	'''
	'''
	return pd.cut(df[var_of_interest], bins, labels = labels)


#### BUILD CLASSIFIER ####
def build_knn(num_neighbors, weights = None, metric = None, metric_params=None):
	'''
	'''
	model = KNeighborsClassifier(n_neighbors=num_neighbors, weights=weights, 
		metric=metric, metric_params=metric_params)
	
	return model

def build_tree():
	'''
	'''
	pass

def proba_wrap(model, x_data, predict = False, threshold = 0.5):
	'''
	'''
	return model.predict_proba(x_data)


#### EVALUATE CLASSIFIER ####
def check_accuracy(y_actual, y_predict):
	'''
	'''
	return metrics.accuracy_score(y_actual, y_predict)

def precision(y_actual, y_predict):
	'''
	'''
	return metrics.precision_score(y_actual, y_predict)

def confusion_matrix(y_actual, y_predict):
	'''
	'''
	# add additional parametres for lables, weight?
	return metrics.confusion_matrix(y_actual, y_predict, labels=None, sample_weight=None)

def classification_report(y_actual, y_predict):
	'''
	'''
	return metrics.classification_report(y_actual, y_predict)

def knn_evaluation_matrix(k_range, x_train, y_train, x_test, y_test, 
	metrics = ['minkowski','euclidean', 'manhattan'], weight_funcs = ['uniform', 'distance']):
	'''
	'''
	df = pd.DataFrame(columns = ['num_neighbors', 'metric', 
		'weighting_function', 'training_acc', 'test_acc', 'train_confusion', 'test_confusion'])
	i = 0

	for k in k_range:
	    for metric in metrics:
	        for func in weight_funcs:
	            params = [k, metric, func]
	            model = build_knn(k, func, metric)
	            model.fit(x_train, y_train)
	            train_pred = model.predict(x_train)
	            test_pred = model.predict(x_test)
	            train_acc = check_accuracy(y_train, train_pred)
	            test_acc = check_accuracy(y_test, test_pred)
	            train_confusion = confusion_matrix(y_train, train_pred)
	            test_confusion = confusion_matrix(y_test, test_pred)
	            tup = [k, metric, func, train_acc, test_acc, train_confusion, test_confusion]
	            df.loc[i] = tup
	            i += 1

	return df

def find_nearest_neighbors():
	'''
	'''
	pass


