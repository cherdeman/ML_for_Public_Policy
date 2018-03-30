import pandas as pd
from sodapy import Socrata
import matplotlib.pyplot as plt
import seaborn as sns

graffiti_id = "cdmx-wzbz"
building_id = "yama-9had"
alley_id = "j9pw-ad5p"
community_id = "igwz-8jzy"

date_svc = "date_service_request_was_received"
creation_date = "creation_date"
date_range = " between '2017-01-01T00:00:00.000' and '2017-12-31T00:00:00.000'"

graffiti_lim = 150000
alley_lim = 30000
building_lim = 5000

index_col = 'service_request_number'
comm_index = 'area_numbe'


def load_full_data():
	'''
	'''
	# Define Socrata client
	client = Socrata("data.cityofchicago.org", None)

	# Load basic dataframes
	graffiti_df = basic_load(graffiti_id, client, creation_date, date_range, 
		graffiti_lim, index_col)
	alley_df = basic_load(alley_id, client, creation_date, date_range, 
		alley_lim, index_col)
	building_df = basic_load(building_id, client, date_svc, date_range, 
		building_lim, index_col)
	
	comm_results = client.get(community_id)
	community_df = pd.DataFrame.from_records(comm_results, index=comm_index)

	# Manipulate building_df to match street_address and type of request fields
	building_df['street_address']= (building_df['address_street_number'] + ' ' + 
		building_df['address_street_direction'] + ' ' + 
		building_df['address_street_name'] + ' ' + 
		building_df['address_street_suffix'])
	building_df['type_of_service_request'] = building_df['service_request_type']
	building_df.drop(['address_street_number', 'address_street_direction', 
		'address_street_name', 'address_street_suffix', 'service_request_type'], 
		axis = 1, inplace = True)

	# Concatenate graffiti, bulding, and alley dataframes
	inter_df = pd.concat([graffiti_df, building_df, alley_df])

	# Pull in community name, drop extraneous columns
	full_df = inter_df.merge(community_df, left_on='community_area', right_index=True)
	full_df.drop(['area', 'area_num_1', 'comarea', 'comarea_id', 'perimeter', 
		'shape_area', 'shape_len', 'the_geom'], axis = 1, inplace = True)

	# Convert date columns to datetime format
	full_df['creation_date'] = pd.to_datetime(full_df['creation_date'])
	full_df['completion_date'] = pd.to_datetime(full_df['completion_date'])
	full_df['date_service_request_was_received'] = pd.to_datetime(
		full_df['date_service_request_was_received'])
	
	# Create response time column
	full_df['response_time'] = (full_df['completion_date'] - 
		full_df['creation_date'])

	# Extract year and month from date columns, unify across datasets
	full_df['year'] = full_df['creation_date'].dt.year
	full_df['year'].fillna(full_df['date_service_request_was_received'].dt.year, 
		inplace = True)
	full_df['month'] = full_df['creation_date'].dt.month
	full_df['month'].fillna((full_df['date_service_request_was_received']
		.dt.month), inplace = True)

	return full_df


def basic_load(dataset_id, client, date_field, date_range, limit_val, index_col):
	'''
	'''
	results = client.get(dataset_id, where=date_field + date_range, 
		limit=limit_val)
	result_df = pd.DataFrame.from_records(results, index=index_col)

	return result_df





											