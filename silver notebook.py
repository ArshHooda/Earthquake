# Databricks notebook source


# COMMAND ----------

import json

# Retrieve the bronze_params directly as a widget
bronze_params = dbutils.widgets.get("bronze_params")
print(f"Raw bronze_params: {bronze_params}")

# Parse the JSON string
output_data = json.loads(bronze_params)

# Access individual variables
start_date = output_data.get("start_date", "")
end_date = output_data.get("end_date", "")
bronze_adls = output_data.get("bronze_adls", "")
silver_adls = output_data.get("silver_adls", "")
gold_adls = output_data.get("gold_adls", "")

print(f"Start Date: {start_date}, Bronze ADLS: {bronze_adls}")

# COMMAND ----------

from pyspark.sql.functions import col, isnull, when
from pyspark.sql.types import TimestampType
from datetime import date, timedelta

# COMMAND ----------

# Load the JSON data into a Spark DataFrame
df = spark.read.option("multiline", "true").json(f"{bronze_adls}{start_date}_earthquake_data.json")

# COMMAND ----------

df

# COMMAND ----------

df.head()

# COMMAND ----------

df = (
    df
    .select(
        'id',
        col('geometry.coordinates').getItem(0).alias('longitude'),
        col('geometry.coordinates').getItem(1).alias('latitude'),
        col('geometry.coordinates').getItem(2).alias('elevation'),
        col('properties.title').alias('title'),
        col('properties.place').alias('place_description'),
        col('properties.sig').alias('sig'),
        col('properties.mag').alias('mag'),
        col('properties.magType').alias('magType'),
        col('properties.time').alias('time'),
        col('properties.updated').alias('updated')
    )
)

# COMMAND ----------

df

# COMMAND ----------

df.head()

# COMMAND ----------

# Validate data: Check for missing or null values
df = (
    df
    .withColumn('longitude', when(isnull(col('longitude')), 0).otherwise(col('longitude')))
    .withColumn('latitude', when(isnull(col('latitude')), 0).otherwise(col('latitude')))
    .withColumn('time', when(isnull(col('time')), 0).otherwise(col('time')))
)

# COMMAND ----------

df

# COMMAND ----------

# Convert 'time' and 'updated' to timestamp from Unix time
df = (
    df
    .withColumn('time', (col('time') / 1000).cast(TimestampType()))
    .withColumn('updated', (col('updated') / 1000).cast(TimestampType()))
)

# COMMAND ----------

df.head()

# COMMAND ----------

# Save the transformed DataFrame to the Silver container
silver_output_path = f"{silver_adls}earthquake_events_silver/"

# COMMAND ----------

# Append DataFrame to Silver container in Parquet format
df.write.mode('append').parquet(silver_output_path)

# COMMAND ----------

dbutils.notebook.exit(silver_output_path)