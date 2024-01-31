from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf,spark_partition_id,monotonically_increasing_id,array,collect_list,size,avg,col,datediff,dayofweek,element_at,split
from pyspark.sql.types import ArrayType,IntegerType
from math import sqrt
from operator import add

# from datetime import date,datetime

# Create instances
sc = SparkContext(appName="PROX-FLIGHTS")
sc.setLogLevel("ERROR")
spark = SparkSession \
    .builder \
    .appName("PROX-FLIGHTS") \
    .getOrCreate()

#big data
df = spark.read.option("header",True).csv("/user/s2595184/flights_dataset.csv")
df.printSchema()

# variables needed!
df2 = df.select(col("legId"),df['searchDate'],df['flightDate'], col("baseFare"),col("segmentsDistance"),col("totalTravelDistance"),(df['baseFare']/col("totalTravelDistance")).alias("pricePerMile"), df['isBasicEconomy'], col("isNonStop"))

# change price to price per mile:
print("all variables!")
df2.show()

print("removing null:")
df2 = df2.filter(col("pricePerMile").isNotNull())
df2.show()
# if distance is Null then  segments Distance is also null!


#Dayofweek: Extract the day of the week of a given date/timestamp as integer. Ranges from 1 for a Sunday through to 7 for a Saturday

df3 = df2.withColumn("daySearch", dayofweek('searchDate'))
df3 = df3.withColumn("dayFlight", dayofweek('flightDate'))
print("showcase Days of search and flight")
df3.show(n=10)

#calculate avg price per daySearch and for dayFlight
df4 = df3.groupBy(col("daySearch"),col("isBasicEconomy"),col("isNonStop")).agg(avg(col('pricePerMile')).alias('pricePerMile')).orderBy(col("isBasicEconomy"),col("isNonStop"),col("daySearch"))
print("showcase avg pricePerMile and daySearch")
df4.show(n=28)

# df4.coalesce(1).write.format("csv").mode('overwrite').save("/user/s2532409/dateSearch_Mile")

# #calculate avg price per daySearch and for dayFlight
df4 = df3.groupBy(col("dayFlight"),col("isBasicEconomy"),col("isNonStop")).agg(avg(col('pricePerMile')).alias('pricePerMile')).orderBy(col("isBasicEconomy"),col("isNonStop"),col("dayFlight"))
print("showcase avg price and dayFlight")
df4.show(n=28)

# df4.coalesce(1).write.format("csv").mode('overwrite').save("/user/s2532409/dateFlight_Mile")


