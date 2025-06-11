# Databricks notebook source
# Mount ADLS Gen2
# Required each time the cluster is restarted which should be only on the first notebook as they run in order

tiers = ["bronze", "silver", "gold"]
adls_paths = {tier: f"abfss://{tier}@dataproject123.dfs.core.windows.net/" for tier in tiers}

# Accessing paths
bronze_adls = adls_paths["bronze"]
silver_adls = adls_paths["silver"]
gold_adls = adls_paths["gold"] 

dbutils.fs.ls(bronze_adls)
dbutils.fs.ls(silver_adls)
dbutils.fs.ls(gold_adls)

# COMMAND ----------

import requests
import json
from datetime import date, timedelta

# COMMAND ----------

dbutils.widgets.text("start_date", "")
dbutils.widgets.text("end_date", "")
start_date = dbutils.widgets.get("start_date")
end_date = dbutils.widgets.get("end_date")

# COMMAND ----------

# Construct the API URL with start and end dates provided by Data Factory, formatted for geojson output.
url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime={start_date}&endtime={end_date}"

try:
    # Make the GET request to fetch data
    response = requests.get(url)

    # Check if the request was successful
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    data = response.json().get('features', [])

    if not data:
        print("No data returned for the specified date range.")
    else:
        # Specify the ADLS path
        file_path = f"{bronze_adls}/{start_date}_earthquake_data.json"

        # Save the JSON data
        json_data = json.dumps(data, indent=4)
        dbutils.fs.put(file_path, json_data, overwrite=True)
        print(f"Data successfully saved to {file_path}")
except requests.exceptions.RequestException as e:
    print(f"Error fetching data from API: {e}")

# COMMAND ----------

data[0]

# COMMAND ----------

# Define your variables
output_data = {
    "start_date": start_date,
    "end_date": end_date,
    "bronze_adls": bronze_adls,
    "silver_adls": silver_adls,
    "gold_adls": gold_adls
}

# Serialize the dictionary to a JSON string
output_json = json.dumps(output_data)

# Log the serialized JSON for debugging
print(f"Serialized JSON: {output_json}")

# Return the JSON string
dbutils.notebook.exit(output_json)