import numpy as np
import pandas as pd
import os

path = 'data/credit-data.csv'
index_col = 'PersonID'

def load_data(path, index_col = None):
	'''
	'''
	if os.path.exists(path):
	    df = pd.read_csv(path)
	else:
		raise Exception('The file does not exist at this location')

	return df


#### EXPLORE DATA ####


def pre_process(df):
	'''
	'''
	# Find columns with missing values
	isnull = df.isnull().any()
	isnull_cols = list(isnull[isnull == True].index)

	for col in isnull_cols:
		col_mean = df[col].mean()
		df[col].fillna(col_mean, inplace = True)

	return df


#### GENERATE FEATURES/PREDICTIONS ####

#### BUILD CLASSIFIER ####

#### EVALUATE CLASSIFIER ####