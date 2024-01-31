from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, dayofyear, dayofweek, col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator




# Create a SparkSession
spark = SparkSession.builder.appName("FeatureImportance").getOrCreate()

# Load the dataset into a PySpark DataFrame
file_path = "./Data/itineraries.csv"  
flight_prices_df = spark.read.csv(file_path, header=True, inferSchema=True)

print("DataSet Loaded")

"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
"""REMOVE FOR FULL DATA PROCESSING"""
# taking sample from dataset
flight_prices_df = flight_prices_df.sample(fraction=0.005, seed=42)


"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""






# print end of processing a dataframe column
def print_process_end(name_column):
    print(f"Processing complete for {name_column}")


# Relevant features list. Add features during the processing
selected_features = []



### Process flightDate ------------------------------------------------------------------------

# Adding new columns
flight_prices_df = flight_prices_df.withColumn("flightDate_YEAR", year("flightDate")) \
                 .withColumn("flightDate_DAYOFYEAR", dayofyear("flightDate")) \
                 .withColumn("flightDate_MONTH", month("flightDate")) \
                 .withColumn("flightDate_DAYOFMONTH", dayofmonth("flightDate")) \
                 .withColumn("flightDate_DAYOFWEEK", dayofweek("flightDate"))

# Add new features to the selected features list
selected_features+= ["flightDate_YEAR", "flightDate_DAYOFYEAR", "flightDate_MONTH", "flightDate_DAYOFMONTH", "flightDate_DAYOFWEEK"]

print_process_end("flightDate")


### Process searchDate ------------------------------------------------------------------------

# Adding new columns
flight_prices_df = flight_prices_df.withColumn("searchDate_YEAR", year("flightDate")) \
                 .withColumn("searchDate_DAYOFYEAR", dayofyear("searchDate")) \
                 .withColumn("searchDate_MONTH", month("searchDate")) \
                 .withColumn("searchDate_DAYOFMONTH", dayofmonth("searchDate")) \
                 .withColumn("searchDate_DAYOFWEEK", dayofweek("searchDate"))

# Add new features to the selected features list
selected_features+= ["searchDate_YEAR", "searchDate_DAYOFYEAR", "searchDate_MONTH", "searchDate_DAYOFMONTH", "searchDate_DAYOFWEEK"]

print_process_end("searchDate")



### Process startingAirport ------------------------------------------------------------------------

# List of unique airports
unique_airports = flight_prices_df.select("startingAirport").distinct().collect()
unique_airports = [row.startingAirport for row in unique_airports]

# Create new columns for each unique airport
for airport in unique_airports:
    new_column_name = f"startingAirport_{airport}"
    flight_prices_df = flight_prices_df.withColumn(new_column_name, when(col("startingAirport") == airport, 1).otherwise(0))
    selected_features.append(new_column_name)

print_process_end("startingAirport")



### Process destinationAirport ------------------------------------------------------------------------

# List of unique airports
unique_airports = flight_prices_df.select("destinationAirport").distinct().collect()
unique_airports = [row.destinationAirport for row in unique_airports]

# Create new columns for each unique airport
for airport in unique_airports:
    new_column_name = f"destinationAirport_{airport}"
    flight_prices_df = flight_prices_df.withColumn(new_column_name, when(col("destinationAirport") == airport, 1).otherwise(0))
    selected_features.append(new_column_name)

print_process_end("destinationAirport")




### Process other features ------------------------------------------------------------------------

selected_features += ["isBasicEconomy", "isRefundable", "isNonStop", "baseFare", "totalTravelDistance"]
selected_features = list(set(selected_features))

# Select the data with relevant features
selected_data = flight_prices_df.select(*selected_features)

# Drop rows with missing values
selected_data = selected_data.dropna()

print_process_end("All Features")



### RANDOM FOREST ------------------------------------------------------------------------

print("Starting Random Forest...")



# Define the feature columns
feature_columns = [col for col in selected_data.columns if col != 'baseFare']

# Create a VectorAssembler to combine features into a single vector column
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Instantiate the RandomForestRegressor model
rf = RandomForestRegressor(featuresCol="features", labelCol="baseFare", numTrees=50, seed=42)

# Create a pipeline to assemble features and train the Random Forest model
pipeline = Pipeline(stages=[assembler, rf])

# Split the data into training and testing sets
train_data, test_data = selected_data.randomSplit([0.8, 0.2], seed=42)

print("Starting Training...")

# Train the model
model = pipeline.fit(train_data)

print("Training ended...")

# Get feature importances
importances = model.stages[-1].featureImportances.toArray()

# Retrieve feature names
feature_names = feature_columns

# Display feature importances
print("Feature Importances:")
for i, imp in enumerate(importances):
    print(f"Feature {feature_names[i]}: {imp}")

# Make predictions
predictions = model.transform(test_data)

# Evaluate the model
mae_evaluator = RegressionEvaluator(labelCol="baseFare", predictionCol="prediction", metricName="mae")
mae = mae_evaluator.evaluate(predictions)
print(f"Mean Absolute Error: {mae}")

mse_evaluator = RegressionEvaluator(labelCol="baseFare", predictionCol="prediction", metricName="mse")
mse = mse_evaluator.evaluate(predictions)
print(f"Mean Squared Error: {mse}")

r2_evaluator = RegressionEvaluator(labelCol="baseFare", predictionCol="prediction", metricName="r2")
r2 = r2_evaluator.evaluate(predictions)
print(f"R-squared: {r2}")

