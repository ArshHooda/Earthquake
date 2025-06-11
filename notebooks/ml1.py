# Databricks notebook source
# JDBC connection to Azure SQL
jdbc_url = "jdbc:sqlserver://gold-ser123.database.windows.net:1433;database=earthquake-db1"
connection_properties = {
    "user": "ETL@gold-ser123",
    "password": "Arsh@2213",
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

# Read from SQL table
df = spark.read.jdbc(
    url=jdbc_url,
    table="earthquake_events",
    properties=connection_properties
)
display(df)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import when, col

# 1. Convert categorical features (country_code) to numeric
country_indexer = StringIndexer(inputCol="country_code", outputCol="country_idx", handleInvalid="keep")

# 2. Assemble features (magnitude, elevation, country)
feature_assembler = VectorAssembler(
    inputCols=["mag", "elevation", "country_idx"],
    outputCol="features"
)

# 3. Convert target (sig_class) to numeric
label_indexer = StringIndexer(inputCol="sig_class", outputCol="label")

# 4. Define the Random Forest model
rf = RandomForestClassifier(
    numTrees=50,
    maxDepth=5,
    seed=42,
    labelCol="label",
    featuresCol="features"
)

# 5. Create ML Pipeline
ml_pipeline = Pipeline(stages=[country_indexer, feature_assembler, label_indexer, rf])

# 6. Split data (80% train, 20% test)
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 7. Train the model
model = ml_pipeline.fit(train_df)

# 8. Make predictions
predictions = model.transform(test_df)

# 9. Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy:.2%}")

# 10. Apply to full dataset
final_predictions = model.transform(df)

# 11. Map predictions back to labels (Low/Moderate/High)
final_predictions = final_predictions.withColumn(
    "ml_prediction",
    when(col("prediction") == 0, "Low")
     .when(col("prediction") == 1, "Moderate")
     .otherwise("High")
)

display(final_predictions.select("title", "sig_class", "ml_prediction"))

# COMMAND ----------

# Add widget parameters at the top
dbutils.widgets.text("output_path", "/mnt/gold/ml_results", "Output path for predictions")
dbutils.widgets.text("email_results", "true", "Send email flag")

from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import when, col, lit, current_timestamp

# 1. Convert categorical features (country_code) to numeric
country_indexer = StringIndexer(inputCol="country_code", outputCol="country_idx", handleInvalid="keep")

# 2. Assemble features (magnitude, elevation, country)
feature_assembler = VectorAssembler(
    inputCols=["mag", "elevation", "country_idx"],
    outputCol="features"
)

# 3. Convert target (sig_class) to numeric
label_indexer = StringIndexer(inputCol="sig_class", outputCol="label")

# 4. Define the Random Forest model
rf = RandomForestClassifier(
    numTrees=50,
    maxDepth=5,
    seed=42,
    labelCol="label",
    featuresCol="features"
)

# 5. Create ML Pipeline
ml_pipeline = Pipeline(stages=[country_indexer, feature_assembler, label_indexer, rf])

# 6. Split data (80% train, 20% test)
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 7. Train the model
model = ml_pipeline.fit(train_df)

# 8. Make predictions
predictions = model.transform(test_df)

# 9. Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy:.2%}")

# 10. Apply to full dataset
final_predictions = model.transform(df)

# 11. Map predictions back to labels (Low/Moderate/High)
final_predictions = final_predictions.withColumn(
    "ml_prediction",
    when(col("prediction") == 0, "Low")
     .when(col("prediction") == 1, "Moderate")
     .otherwise("High")
)
display(final_predictions.select("title", "sig_class", "ml_prediction"))


import json

# 1. Save predictions
output_path = dbutils.widgets.get("output_path")
(final_predictions
 .withColumn("processing_time", lit(current_timestamp()))
 .write.mode("overwrite")
 .parquet(output_path))

# 2. Prepare email content
email_data = {
    "timestamp": str(current_timestamp()),
    "accuracy": float(accuracy),
    "high_risk_events": [row.asDict() for row in 
        final_predictions.filter(col("ml_prediction") == "High")
        .select("title", "mag", "place_description")
        .limit(5).collect()],
    "discrepancies": final_predictions.filter(col("sig_class") != col("ml_prediction")).count(),
    "prediction_path": output_path
}

# 3. Return to Data Factory
dbutils.notebook.exit(json.dumps(email_data))