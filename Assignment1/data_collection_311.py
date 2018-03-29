import pandas as pd
from sodapy import Socrata

graffiti_dataset_id = "cdmx-wzbz"
buildings_dataset_id = "yama-9had"
alley_dataset_id = "j9pw-ad5p"
building_date_range = "date_service_request_was_received between '2017-01-01T00:00:00.000' and '2018-01-01T00:00:00.000'"
graffiti_alley_date_range = "creation_date between '2017-01-01T00:00:00.000' and '2018-01-01T00:00:00.000'"

# Define Socrata client
client = Socrata("data.cityofchicago.org", None)

											#"ztMpoYxFHVD1jdYwWoBpEI6IO", 
											#username="cherdeman@uchicago.edu", 
											#password="SzQx8*CIC$zX")

# First 2000 results, returned as JSON from API / converted to Python list of
# dictionaries by sodapy.
graffiti_results = client.get(graffiti_dataset_id, where=graffiti_alley_date_range, limit = 150000)
building_results = client.get(buildings_dataset_id, where=building_date_range, limit = 5000)
alley_results = client.get(alley_dataset_id, where=graffiti_alley_date_range, limit = 30000)

# Convert to pandas DataFrame
graffiti_df = pd.DataFrame.from_records(graffiti_results)
building_df = pd.DataFrame.from_records(building_results)
alley_df = pd.DataFrame.from_records(alley_results)

# determine columns for merge
intersect = [x for x in graffiti_df if x in building_df and  x in alley_df]

result = pd.concat([graffiti_df, building_df, alley_df])
