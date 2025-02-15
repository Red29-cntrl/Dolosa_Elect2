import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
from pyspark.ml.feature import StringIndexer

# *** CRITICAL: Environment variables MUST be set BEFORE creating the SparkSession ***
# 1. **Absolute Hadoop Version Match:**
#    - Find your Spark version: print(spark.version)  (do this *after* creating SparkSession if needed)
#    - Find the Hadoop version Spark uses (check Spark docs for your version).
#    - Download winutils.exe and DLLs *precisely* matching that Hadoop version.
hadoop_home = "C:/winutils"  # *** Replace with the *correct* path ***
os.environ["HADOOP_HOME"] = hadoop_home
os.environ["PATH"] = os.path.join(hadoop_home, "bin") + ";" + os.environ["PATH"]
os.environ["SPARK_LOCAL_DIRS"] = "C:/tmp"  # This is usually okay

# 2. Initialize Spark Session
spark = SparkSession.builder \
    .appName("Insurance Data Preprocessing") \
    .config("spark.sql.warehouse.dir", "file:///C:/tmp") \
    .getOrCreate()

# Load dataset
insurance_df = spark.read.csv("insurance.csv", header=True, inferSchema=True)

# Display schema and first few rows
insurance_df.printSchema()
insurance_df.show(5)

# Check for missing values
insurance_df.select([count(col(c)).alias(c) for c in insurance_df.columns]).show()

# Fill missing BMI values with the mean BMI
bmi_mean = insurance_df.agg({"bmi": "mean"}).collect()[0][0]
insurance_df = insurance_df.fillna({"bmi": bmi_mean})

# Filter records where BMI is between 15 and 50
insurance_df = insurance_df.filter((col("bmi") >= 15) & (col("bmi") <= 50))

# Convert categorical variables into numerical values
indexer = StringIndexer(inputCol="sex", outputCol="sex_indexed")
insurance_df = indexer.fit(insurance_df).transform(insurance_df)

# Calculate average insurance charges by region
insurance_df.groupBy("region").avg("charges").show()

# Load Smoking Data
file_path = "C:/Users/Win11/Documents/ElectLab3/smoking_data.csv"
smoking_data = spark.read.format("csv").option("header", "true").load(file_path)

# Debug Column Names
print("Smoking Data Columns:", smoking_data.columns)

# Rename if necessary
smoking_data = smoking_data.withColumnRenamed("Smoker", "smoker") if "Smoker" in smoking_data.columns else smoking_data

# Ensure 'smoker' has the same type in both dataframes
insurance_df = insurance_df.withColumn("smoker", col("smoker").cast("string"))
smoking_data = smoking_data.withColumn("smoker", col("smoker").cast("string"))

# Join DataFrames
joint_df = insurance_df.join(smoking_data, "smoker", "inner")

# 3. Save cleaned dataset
output_path = "C:/Users/Win11/Documents/ElectLab3/processed_insurance.csv"
insurance_df.write.mode("overwrite").csv(output_path, header=True)

# 4. Stop Spark Session
spark.stop()