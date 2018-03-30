import pandas as pd
from sodapy import Socrata

graffiti_id = "cdmx-wzbz"
buildings_id = "yama-9had"
alley_id = "j9pw-ad5p"

date_svc = "date_service_request_was_received"
creation_date = "creation_date"
date_range = " between '2017-01-01T00:00:00.000' and '2018-01-01T00:00:00.000'"

graffiti_lim = 150000
alley_lim = 30000
building_lim = 5000


#building_date_range = "date_service_request_was_received between '2017-01-01T00:00:00.000' and '2018-01-01T00:00:00.000'"
#graffiti_alley_date_range = "creation_date between '2017-01-01T00:00:00.000' and '2018-01-01T00:00:00.000'"
#community_areas_id = "igwz-8jzy"

def load_full_data():
	'''
	'''
	# Define Socrata client
	client = Socrata("data.cityofchicago.org", None)

	# Load basic dataframes
	graffiti_df = basic_load(graffiti_id, client, creation_date, date_range, 
		graffiti_lim)
	alley_df = basic_load(alley_id, client, creation_date, date_range, 
		alley_lim)
	building_df = basic_load(building_id, client, date_svc, date_range, 
		building_lim)

	# Manipulate building_df to match street_address and type of request fields
	building_df['street_address']= (building_df['address_street_number'] + ' ' + 
		building_df['address_street_direction'] + ' ' + 
		building_df['address_street_name'] + ' ' + 
		building_df['address_street_suffix'])
	building_df['type_of_service_request'] = building_df['service_request_type']
	building_df.drop(['address_street_number', 'address_street_direction', 
		'address_street_name', 'address_street_suffix', 'service_request_type'], 
		axis = 1, inplace = True)

	# Merge dataframe
	full_df = pd.concat([graffiti_df, building_df, alley_df])

	# Convert date columns to datetime format
	full_df['creation_date'] = pd.to_datetime(full_df['creation_date'])
	full_df['completion_date'] = pd.to_datetime(full_df['completion_date'])
	full_df['date_service_request_was_received'] = pd.to_datetime(
		full_df['date_service_request_was_received'])
	full_df['response_time'] = (full_df['completion_date'] - 
		full_df['creation_date'])

	full_df['year'] = full_df['creation_date'].dt.year
	full_df['year'].fillna(full_df['date_service_request_was_received'].dt.year, 
		inplace = True)
	full_df['month'] = full_df['creation_date'].dt.month
	full_df['month'].fillna((full_df['date_service_request_was_received']
		.dt.month), inplace = True)

	return full_df


def basic_load(dataset_id, client, date_field, date_range, limit_val):
	'''
	'''
	results = client.get(dataset_id, where=date_field + date_range, limit=limit_val)
	result_df = pd.DataFrame.from_records(results)

	return result_df


											

# First 2000 results, returned as JSON from API / converted to Python list of
# dictionaries by sodapy.
graffiti_results = client.get(graffiti_dataset_id, where=graffiti_alley_date_range, limit = 150000)
building_results = client.get(buildings_dataset_id, where=building_date_range, limit = 5000)
alley_results = client.get(alley_dataset_id, where=graffiti_alley_date_range, limit = 30000)

# Convert to pandas DataFrame
graffiti_df = pd.DataFrame.from_records(graffiti_results)
building_df = pd.DataFrame.from_records(building_results)
alley_df = pd.DataFrame.from_records(alley_results)