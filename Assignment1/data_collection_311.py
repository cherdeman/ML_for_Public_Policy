import numpy as np
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

	full_df['request_count'] = 1

	return full_df


def basic_load(dataset_id, client, date_field, date_range, limit_val, index_col):
	'''
	'''
	results = client.get(dataset_id, where=date_field + date_range, 
		limit=limit_val)
	result_df = pd.DataFrame.from_records(results, index=index_col)

	return result_df

def summary_statistics(df, request_type, aggregator):
	'''
	'''
	summary_table = pd.pivot_table(df,values='request_count', 
		index=aggregator, columns=request_type, aggfunc=np.sum)

	return summary_table

from census import Census
from us import states

my_api_key = "217c49410f1af077f4d96b7d5da3612d06eae18d"
cook_county_fips_id = '031'
fields_of_interest = ('NAME', 'B02001_001E', 'B02001_003E', 'B25003_001E',
	'B25003_002E','B25003_003E','B19001_001E','B19001_002E','B19001_003E',
	'B19001_004E','B19001_005E','B19001_006E')
IL_state_ID = states.IL.fips
field_name_dict = {'B02001_001E': 'Race Total', 'B02001_003E': 'Black Total', 
	'B25003_001E': 'Total Occupied', 'B25003_002E':'Owner Occupied',
	'B25003_003E': 'Renter Occupied','B19001_001E': 'Total Household Income',
	'B19001_002E': '<$10k','B19001_003E': '$10k-$15k', 
	'B19001_004E': '$15k-$20k','B19001_005E':'$20k-$25k',
	'B19001_006E': '$25k-30k'}

def census_data_pull(API_key, fields, state_id, county_id, field_name_dict, block_group_id=Census.ALL):
	'''
	'''
	c = Census(API_key)

	results = c.acs5.state_county_blockgroup(fields, state_id, county_id, block_group_id)
	results_df = pd.DataFrame.from_records(results)

	results_df.rename(columns = field_name_dict, inplace = True)

	return results_df

'''
'B02001_001E', 'B02001_003E' (race)
B25003_001E	Estimate!!Total	TENURE	not required	B25003_001M, B25003_001MA, B25003_001EA	0	int	B25003	N/A
B25003_002E	Estimate!!Total!!Owner occupied	TENURE	not required	B25003_002M, B25003_002MA, B25003_002EA	0	int	B25003	N/A
B25003_003E

B19051_001E	Estimate!!Total	EARNINGS IN THE PAST 12 MONTHS FOR HOUSEHOLDS	not required	B19051_001M, B19051_001MA, B19051_001EA	0	int	B19051	N/A
B19051_002E	Estimate!!Total!!With earnings	EARNINGS IN THE PAST 12 MONTHS FOR HOUSEHOLDS	not required	B19051_002M, B19051_002MA, B19051_002EA	0	int	B19051	N/A
B19051_003E	Estimate!!Total!!No earnings	EARNINGS IN THE PAST 12 MONTHS FOR HOUSEHOLDS	not required	B19051_003M, B19051_003MA, B19051_003EA	0	int	B19051	N/A
B19052_001E	Estimate!!Total	WAGE OR SALARY INCOME IN THE PAST 12 MONTHS FOR HOUSEHOLDS	not required	B19052_001M, B19052_001MA, B19052_001EA	0	int	B19052	N/A
B19052_002E	Estimate!!Total!!With wage or salary income	WAGE OR SALARY INCOME IN THE PAST 12 MONTHS FOR HOUSEHOLDS	not required	B19052_002M, B19052_002MA, B19052_002EA	0	int	B19052	N/A
B19052_003E	Estimate!!Total!!No wage or salary income

B19001_001E	Estimate!!Total	HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS)	not required	B19001_001M, B19001_001MA, B19001_001EA	0	int	B19001	N/A
B19001_002E	Estimate!!Total!!Less than $10,000	HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS)	not required	B19001_002M, B19001_002MA, B19001_002EA	0	int	B19001	N/A
B19001_003E	Estimate!!Total!!$10,000 to $14,999	HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS)	not required	B19001_003M, B19001_003MA, B19001_003EA	0	int	B19001	N/A
B19001_004E	Estimate!!Total!!$15,000 to $19,999	HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS)	not required	B19001_004M, B19001_004MA, B19001_004EA	0	int	B19001	N/A
B19001_005E	Estimate!!Total!!$20,000 to $24,999	HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS)	not required	B19001_005M, B19001_005MA, B19001_005EA	0	int	B19001	N/A
B19001_006E	Estimate!!Total!!$25,000 to $29,999	HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS)	not required	B19001_006M, B19001_006MA, B19001_006EA	0	int	B19001	N/A
B19001_007E	Estimate!!Total!!$30,000 to $34,999	HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS)	not required	B19001_007M, B19001_007MA, B19001_007EA	0	int	B19001	N/A
B19001_008E	Estimate!!Total!!$35,000 to $39,999	HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS)	not required	B19001_008M, B19001_008MA, B19001_008EA	0	int	B19001	N/A
B19001_009E	Estimate!!Total!!$40,000 to $44,999	HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS)	not required	B19001_009M, B19001_009MA, B19001_009EA	0	int	B19001	N/A
B19001_010E	Estimate!!Total!!$45,000 to $49,999	HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS)	not required	B19001_010M, B19001_010MA, B19001_010EA	0	int	B19001	N/A
B19001_011E	Estimate!!Total!!$50,000 to $59,999	HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS)	not required	B19001_011M, B19001_011MA, B19001_011EA	0	int	B19001	N/A
B19001_012E	Estimate!!Total!!$60,000 to $74,999	HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS)	not required	B19001_012M, B19001_012MA, B19001_012EA	0	int	B19001	N/A
B19001_013E	Estimate!!Total!!$75,000 to $99,999	HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS)	not required	B19001_013M, B19001_013MA, B19001_013EA	0	int	B19001	N/A
B19001_014E	Estimate!!Total!!$100,000 to $124,999	HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS)	not required	B19001_014M, B19001_014MA, B19001_014EA	0	int	B19001	N/A
B19001_015E	Estimate!!Total!!$125,000 to $149,999	HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS)	not required	B19001_015M, B19001_015MA, B19001_015EA	0	int	B19001	N/A
B19001_016E	Estimate!!Total!!$150,000 to $199,999	HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS)	not required	B19001_016M, B19001_016MA, B19001_016EA	0	int	B19001	N/A
B19001_017E	Estimate!!Total!!$200,000 or more	HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS)	not required	B19001_017M, B19001_017MA, B19001_017EA	0	int	B19001	N/A

B15003_001E	Estimate!!Total	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_001M, B15003_001MA, B15003_001EA	0	int	B15003	N/A
B15003_002E	Estimate!!Total!!No schooling completed	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_002M, B15003_002MA, B15003_002EA	0	int	B15003	N/A
B15003_003E	Estimate!!Total!!Nursery school	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_003M, B15003_003MA, B15003_003EA	0	int	B15003	N/A
B15003_004E	Estimate!!Total!!Kindergarten	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_004M, B15003_004MA, B15003_004EA	0	int	B15003	N/A
B15003_005E	Estimate!!Total!!1st grade	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_005M, B15003_005MA, B15003_005EA	0	int	B15003	N/A
B15003_006E	Estimate!!Total!!2nd grade	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_006M, B15003_006MA, B15003_006EA	0	int	B15003	N/A
B15003_007E	Estimate!!Total!!3rd grade	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_007M, B15003_007MA, B15003_007EA	0	int	B15003	N/A
B15003_008E	Estimate!!Total!!4th grade	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_008M, B15003_008MA, B15003_008EA	0	int	B15003	N/A
B15003_009E	Estimate!!Total!!5th grade	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_009M, B15003_009MA, B15003_009EA	0	int	B15003	N/A
B15003_010E	Estimate!!Total!!6th grade	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_010M, B15003_010MA, B15003_010EA	0	int	B15003	N/A
B15003_011E	Estimate!!Total!!7th grade	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_011M, B15003_011MA, B15003_011EA	0	int	B15003	N/A
B15003_012E	Estimate!!Total!!8th grade	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_012M, B15003_012MA, B15003_012EA	0	int	B15003	N/A
B15003_013E	Estimate!!Total!!9th grade	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_013M, B15003_013MA, B15003_013EA	0	int	B15003	N/A
B15003_014E	Estimate!!Total!!10th grade	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_014M, B15003_014MA, B15003_014EA	0	int	B15003	N/A
B15003_015E	Estimate!!Total!!11th grade	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_015M, B15003_015MA, B15003_015EA	0	int	B15003	N/A
B15003_016E	Estimate!!Total!!12th grade, no diploma	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_016M, B15003_016MA, B15003_016EA	0	int	B15003	N/A
B15003_017E	Estimate!!Total!!Regular high school diploma	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_017M, B15003_017MA, B15003_017EA	0	int	B15003	N/A
B15003_018E	Estimate!!Total!!GED or alternative credential	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_018M, B15003_018MA, B15003_018EA	0	int	B15003	N/A
B15003_019E	Estimate!!Total!!Some college, less than 1 year	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_019M, B15003_019MA, B15003_019EA	0	int	B15003	N/A
B15003_020E	Estimate!!Total!!Some college, 1 or more years, no degree	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_020M, B15003_020MA, B15003_020EA	0	int	B15003	N/A
B15003_021E	Estimate!!Total!!Associate's degree	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_021M, B15003_021MA, B15003_021EA	0	int	B15003	N/A
B15003_022E	Estimate!!Total!!Bachelor's degree	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_022M, B15003_022MA, B15003_022EA	0	int	B15003	N/A
B15003_023E	Estimate!!Total!!Master's degree	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_023M, B15003_023MA, B15003_023EA	0	int	B15003	N/A
B15003_024E	Estimate!!Total!!Professional school degree	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER	not required	B15003_024M, B15003_024MA, B15003_024EA	0	int	B15003	N/A
B15003_025E	Estimate!!Total!!Doctorate degree	EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER
'''
											