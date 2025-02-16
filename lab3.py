import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
from pyspark.ml.feature import StringIndexer

# *** Set Up Hadoop Environment ***
hadoop_home = "C:/winutils"  # Ensure this path is correct
os.environ["HADOOP_HOME"] = hadoop_home
os.environ["PATH"] = os.path.join(hadoop_home, "bin") + ";" + os.environ["PATH"]
os.environ["SPARK_LOCAL_DIRS"] = "C:/tmp"  # Usually okay

# *** Initialize Spark Session ***
spark = SparkSession.builder \
    .appName("Insurance Data Preprocessing") \
    .config("spark.sql.warehouse.dir", "file:///C:/tmp") \
    .getOrCreate()

# *** Define File Paths ***
insurance_path = "C:/Users/Win11/Documents/ElectLab3/Dolosa_Elect2/insurance.csv"
smoking_data_path = "C:/Users/Win11/Documents/ElectLab3/Dolosa_Elect2/smoking_data.csv"
output_path = "C:/Users/Win11/Documents/ElectLab3/Dolosa_Elect2/processed_insurance.csv"

# *** Verify File Existence Before Reading ***
if not os.path.exists(insurance_path):
    raise FileNotFoundError(f"Error: Insurance file not found at {insurance_path}")
if not os.path.exists(smoking_data_path):
    raise FileNotFoundError(f"Error: Smoking data file not found at {smoking_data_path}")

# *** Load Insurance Dataset ***
insurance_df = spark.read.csv(insurance_path, header=True, inferSchema=True)

# *** Display Schema and First Few Rows ***
insurance_df.printSchema()
insurance_df.show(5)

# *** Check for Missing Values ***
insurance_df.select([count(col(c)).alias(c) for c in insurance_df.columns]).show()

# *** Fill Missing BMI Values with Mean ***
bmi_mean = insurance_df.agg({"bmi": "mean"}).collect()[0][0]
insurance_df = insurance_df.fillna({"bmi": bmi_mean})

# *** Filter Records Where BMI is Between 15 and 50 ***
insurance_df = insurance_df.filter((col("bmi") >= 15) & (col("bmi") <= 50))

# *** Convert Categorical Variables to Numerical Values ***
indexer = StringIndexer(inputCol="sex", outputCol="sex_indexed")
insurance_df = indexer.fit(insurance_df).transform(insurance_df)

# *** Calculate Average Insurance Charges by Region ***
insurance_df.groupBy("region").avg("charges").show()

# *** Load Smoking Data ***
smoking_data = spark.read.format("csv").option("header", "true").load(smoking_data_path)

# *** Debug Column Names ***
print("Smoking Data Columns:", smoking_data.columns)

# *** Rename 'Smoker' Column if Necessary ***
if "Smoker" in smoking_data.columns:
    smoking_data = smoking_data.withColumnRenamed("Smoker", "smoker")

# *** Ensure 'smoker' Has the Same Type in Both DataFrames ***
insurance_df = insurance_df.withColumn("smoker", col("smoker").cast("string"))
smoking_data = smoking_data.withColumn("smoker", col("smoker").cast("string"))

# *** Join DataFrames on 'smoker' Column ***
joint_df = insurance_df.join(smoking_data, "smoker", "inner")

# *** Save Cleaned Dataset ***
joint_df.write.mode("overwrite").csv(output_path, header=True)

# *** Stop Spark Session ***
spark.stop()

# *** May aayusin pa huhu***
