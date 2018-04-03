import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from sodapy import Socrata
import matplotlib.pyplot as plt
import seaborn as sns
from census import Census
from us import states

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
	Load Graffiti Removal, Vacant/Abandoned Buildings, Alley Light Out requests
	from Chicago Open Portal into pandas datafarme

	Inputs: None, relies on global variables as defined

	Returns: Pandas dataframe
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
	full_df = inter_df.merge(community_df, left_on='community_area', 
		right_index=True)
	full_df.drop(['area', 'area_num_1', 'comarea', 'comarea_id', 'perimeter', 
		'shape_area', 'shape_len', 'the_geom', 'x_coordinate', 'y_coordinate'], 
		axis = 1, inplace = True)

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

	full_df['response_time'] = full_df['response_time'].astype(
		'timedelta64[ms]') / 86400000

	full_df['request_count'] = 1

	return full_df


def basic_load(dataset_id, client, date_field, date_range, limit_val, index_col):
	'''
	Basic loading function to call to Chicago Open Portal and return a dataframe

	Inputs:
		-dataset_id (str): 9 digit key to call 
		-client (Socrata object): client to use to call to API
		-date_field (str): defines column containing date in given dataset
		-date_range (str): defines data range of interest
		-limit_val (int): maximum number of recors to pull
		-index_col (str): column to define as index

	Returns a pandas dataframe
	'''
	results = client.get(dataset_id, where=date_field + date_range, 
		limit=limit_val)
	result_df = pd.DataFrame.from_records(results, index=index_col)

	return result_df

def summary_statistics(df, request_type, aggregator, aggfunc):
	'''
	Output a pandas pivot table to represent basic summary statistics

	Inputs:
		- df (dataframe): dataframe to manipulate
		- request_type (str): column of interest
		- aggregator (str): column to use to aggregate data
		- aggfunc (numpy function): function to use in aggregation

	Returns pandas pivot table
	'''
	summary_table = pd.pivot_table(df,values='request_count', 
		index=aggregator, columns=request_type, aggfunc=aggfunc)

	return summary_table


############################# Census Data Loading #############################

my_api_key = "217c49410f1af077f4d96b7d5da3612d06eae18d"
fields_of_interest = ('NAME', 'B02001_001E', 'B02001_003E', 'B25003_001E',
	'B25003_002E','B25003_003E','B19001_001E','B19001_002E','B19001_003E',
	'B19001_004E','B19001_005E','B19001_006E')
IL_state_ID = states.IL.fips
cook_county_fips_id = '031'
field_name_dict = {'B02001_001E': 'race_total', 'B02001_003E': 'black_total', 
	'B25003_001E': 'total_occupied', 'B25003_002E':'owner_occupied',
	'B25003_003E': 'renter_occupied','B19001_001E': 'total_household_income',
	'B19001_002E': '<$10k','B19001_003E': '$10k-$15k', 
	'B19001_004E': '$15k-$20k','B19001_005E':'$20k-$25k',
	'B19001_006E': '$25k-30k'}
block_shape_path = './cb_2016_17_bg_500k/cb_2016_17_bg_500k.shp'


def census_data_pull(API_key, fields, state_id, county_id, field_name_dict, 
	block_group_id=Census.ALL):
	'''
	Pull census data from ACS 5 year using API

	Inputs:
		- API_key (str): census API key
		- fields (set): fields to pull from Census API
		- state_id : state fips code as defined by US package
		- county_id (str): County fips code 
		- field_name_dict (dict): mapping of field codes to field names
		- block_group_id (str): block group fips code, set to ALL if unspecified

	Returns a GeoPandas GeoDataframe
	'''
	c = Census(API_key)

	results = c.acs5.state_county_blockgroup(fields, state_id, county_id, block_group_id)
	results_gdf = gpd.GeoDataFrame.from_records(results)

	results_gdf.rename(columns = field_name_dict, inplace = True)

	return results_gdf


def census_calcs(census_gdf):
	'''
	Data manipulation on census dataframe

	Inputs:
		- census_gdf (GeoDataFrame): dataframe resulting from previous function

	Returns an updated GeoDataFrame
	'''
	pct_black = census_gdf.loc[:,('black_total')] / census_gdf.loc[:,('race_total')]
	pct_owner = census_gdf.loc[:,('owner_occupied')] / census_gdf.loc[:,('total_occupied')]
	pct_renter = census_gdf.loc[:, ('renter_occupied')] / census_gdf.loc[:,('total_occupied')]
	pct_under_30k = census_gdf[['<$10k', '$10k-$15k', '$15k-$20k','$20k-$25k',
	'$25k-30k']].sum(axis = 1) / census_gdf.loc[:,('total_household_income')]

	census_gdf = census_gdf.assign(pct_black = pct_black)
	census_gdf = census_gdf.assign(pct_owner = pct_owner)
	census_gdf = census_gdf.assign(pct_under_30k = pct_under_30k)

	census_gdf.drop(['black_total','owner_occupied','renter_occupied','<$10k', 
		'$10k-$15k', '$15k-$20k','$20k-$25k','$25k-30k','race_total',
		'total_occupied','total_household_income'], axis = 1, inplace = True)

	census_gdf['GEOID'] = (census_gdf['state'] + census_gdf['county'] 
		+ census_gdf['tract'] + census_gdf['block group'])

	return census_gdf


def merge_census_shape_dfs(census_gdf, shape_path):
	'''
	Merge census GeoDataFrame with a shape file to connect spatial data

	Inputs:
		- census_gdf (GeoDataFrame): dataframe resulting from previous function
		- shape_path (str): represents location of shape file on local machine

	Returns an updated GeoDataFrame
	'''
	shape_gdf = gpd.read_file(shape_path)
	shape_gdf.drop('NAME', axis = 1, inplace = True)

	merged_df = census_gdf.merge(shape_gdf, on = 'GEOID')
	crs = {'init': 'epsg:4326'}
	merged_gdf = GeoDataFrame(merged_df, crs=crs)
	merged_gdf = merged_gdf.to_crs(crs)

	return merged_gdf

def spatial_join(full_df, census_gdf):
	'''
	Join the merged census dataframe with the previous full dataframe

	Inputs:
		- full_df (dataframe): dataframe with a point geography
		- census_gdf (GeoDataFrame): dataframe resulting from previous function

	Returns full_df, augmented with census data based on the spatial join
	'''
	full_df[['latitude','longitude']] = full_df[['latitude','longitude']].apply(pd.to_numeric)
	geometry = [Point(xy) for xy in zip(full_df['longitude'], full_df['latitude'])]
	full_df = full_df.drop(['latitude', 'longitude'], axis=1)
	crs = {'init': 'epsg:4326'}
	full_gdf = GeoDataFrame(full_df, crs=crs, geometry=geometry)
	full_gdf = full_gdf[full_gdf['geometry'].is_valid]
	full_gdf = full_gdf.to_crs(crs)

	full_with_spatial = gpd.sjoin(full_gdf, census_gdf, how="inner", op='intersects')

	return full_with_spatial

