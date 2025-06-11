# Databricks notebook source
from datetime import date, timedelta

# Remove this before running Data Factory Pipeline
start_date = date.today() - timedelta(1)

silver_adls = "abfss://silver@dataproject123.dfs.core.windows.net/"
gold_adls = "abfss://gold@dataproject123.dfs.core.windows.net/"

silver_data = f"{silver_adls}earthquake_events_silver/"

# COMMAND ----------

'''import json

# Get base parameters
dbutils.widgets.text("bronze_params", "")
dbutils.widgets.text("silver_params", "")

bronze_params = dbutils.widgets.get("bronze_params")
silver_params = dbutils.widgets.get("silver_params")

# Debug: Print the raw input values for troubleshooting
print(f"Raw bronze_params: {bronze_params}")
print(f"Raw silver_params: {silver_params}")

# Parse the JSON string if not empty
if bronze_params:
    bronze_data = json.loads(bronze_params)
else:
    bronze_data = {}

# Access individual variables
start_date = bronze_data.get("start_date", "")
end_date = bronze_data.get("end_date", "")
silver_adls = bronze_data.get("silver_adls", "")
gold_adls = bronze_data.get("gold_adls", "")
silver_data = silver_params

# Debug: Print the extracted values for verification
print(f"Start Date: {start_date}, End Date: {end_date}")
print(f"Silver ADLS Path: {silver_adls}, Gold ADLS Path: {gold_adls}")'''

# COMMAND ----------

from pyspark.sql.functions import when, col, udf
from pyspark.sql.types import StringType
# Ensure the below library is installed on your cluster
import reverse_geocoder as rg
from datetime import date, timedelta

# COMMAND ----------

from pyspark.sql.functions import col

silver_data = "abfss://silver@dataproject123.dfs.core.windows.net/earthquake_events_silver/"
start_date = "2025-05-29"

df = spark.read.parquet(silver_data).filter(col('time') > start_date)
display(df)

# COMMAND ----------

df = df.limit(200) # added to speed up processings as during testing it was proving a bottleneck
# The problem is caused by the Python UDF (reverse_geocoder) being a bottleneck due to its non-parallel nature and high computational cost per task
     

# COMMAND ----------

def get_country_code(lat, lon):
    """
    Retrieve the country code for a given latitude and longitude.

    Parameters:
    lat (float or str): Latitude of the location.
    lon (float or str): Longitude of the location.

    Returns:
    str: Country code of the location, retrieved using the reverse geocoding API.

    Example:
    >>> get_country_details(48.8588443, 2.2943506)
    'FR'
    """
    try:
        coordinates = (float(lat), float(lon))
        result = rg.search(coordinates)[0].get('cc')
        print(f"Processed coordinates: {coordinates} -> {result}")
        return result
    except Exception as e:
         print(f"Error processing coordinates: {lat}, {lon} -> {str(e)}")
         return None

# COMMAND ----------

# registering the udfs so they can be used on spark dataframes
get_country_code_udf = udf(get_country_code, StringType())

# COMMAND ----------

get_country_code(48.8588443, 2.2943506)

# COMMAND ----------

# adding country_code and city attributes
df_with_location = \
                df.\
                    withColumn("country_code", get_country_code_udf(col("latitude"), col("longitude")))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df_with_location.printSchema()

# COMMAND ----------

# adding significance classification
df_with_location_sig_class = \
                            df_with_location.\
                                withColumn('sig_class', 
                                            when(col("sig") < 100, "Low").\
                                            when((col("sig") >= 100) & (col("sig") < 500), "Moderate").\
                                            otherwise("High")
                                            )

# COMMAND ----------

df_with_location_sig_class.printSchema()

# COMMAND ----------

# Save the transformed DataFrame to the Silver container
gold_output_path = f"{gold_adls}earthquake_events_gold/"

# COMMAND ----------

# Append DataFrame to Silver container in Parquet format
df_with_location_sig_class.write.mode('append').csv(gold_output_path)

# COMMAND ----------

df_with_location_sig_class.display()

# COMMAND ----------

gold_output_path

# COMMAND ----------

from pyspark.sql.functions import col

float_columns = ["longitude", "latitude", "elevation", "mag"]

for col_name in float_columns:
    df_cleaned = df_cleaned.withColumn(col_name, col(col_name).cast(FloatType()))


# Define JDBC connection string for Azure SQL Database
jdbc_url = "jdbc:sqlserver://gold-ser123.database.windows.net:1433;" \
           "database=earthquake-db1;" \
           "user=ETL@gold-ser123;" \
           "password=Arsh@2213;" \
           "encrypt=true;" \
           "trustServerCertificate=false;" \
           "hostNameInCertificate=*.database.windows.net;" \
           "loginTimeout=30;" \
           "connectTimeout=30;" \
           "socketTimeout=30;" 

# Drop rows with null sig_class to meet NOT NULL constraint
df_cleaned = df_with_location_sig_class.filter("sig_class IS NOT NULL")

# Write DataFrame to the existing SQL table with error handling
try:
    df_cleaned.write \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", "earthquake_events") \
        .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver") \
        .option("batchsize", 10000) \
        .mode("append") \
        .save()
    print("✅ Data successfully written to earthquake_events table")
except Exception as e:
    print(f"❌ Error writing to database: {str(e)}")