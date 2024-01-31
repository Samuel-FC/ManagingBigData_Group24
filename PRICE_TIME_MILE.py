from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg,col,to_timestamp,split,element_at,hour,substring,length,expr,count
from pyspark.sql.types import ArrayType,IntegerType
from math import sqrt
from operator import add

# from datetime import date,datetime

# Create instances
sc = SparkContext(appName="PRICE_TIME")
sc.setLogLevel("ERROR")
spark = SparkSession \
    .builder \
    .appName("PRICE_TIME") \
    .getOrCreate()

#big data
df = spark.read.option("header",True).csv("/user/s2595184/flights_dataset.csv")
df.printSchema()

# variables needed!
df2 = df.select(col("legId"),(df['baseFare']/col("totalTravelDistance")).alias("pricePerMile"), df['isBasicEconomy'], col("isNonStop"),col("totalTravelDistance"), \
                col("segmentsDepartureTimeRaw").alias("ts_raw"))
print("general info")
df2.show(truncate=50)

# filter all the distances with null:
print("removing null for distance:")
df2 = df2.filter(col("pricePerMile").isNotNull())
df2.show()


#gather the times
# https://www3.ntu.edu.sg/home/ehchua/programming/howto/Regexe.html
df2 = df2.withColumn("ts_raw", element_at(split(col("ts_raw"),"\|\|"), 1))
df2 = df2.withColumn("ts_raw", expr("substring(ts_raw,1,length(ts_raw)-6)"))
# df2.show(truncate=100)
df2 = df2.withColumn("ts_raw", to_timestamp(col("ts_raw")))
df2.show(truncate=100)
df2 = df2.withColumn("hours", hour(col("ts_raw")))
# df2 = df2.withColumn("hours", hour(col("ts_raw")))
df2.show(truncate=100)

# take avg of all the flights and searchdates!
df3 = df2.groupBy(col("hours"), col("isBasicEconomy"),col("isNonStop")).agg(avg(col("pricePerMile")).alias("pricePerMile"), count(col("pricePerMile")),avg(col("totalTravelDistance"))).\
    orderBy(col("isBasicEconomy"),col("isNonStop"),col("hours"))

df3.show(n=96)
# +-----+--------------+---------+-------------------+-------------------+------------------------+
# |hours|isBasicEconomy|isNonStop|       pricePerMile|count(pricePerMile)|avg(totalTravelDistance)|
# +-----+--------------+---------+-------------------+-------------------+------------------------+
# |    0|         False|    False|0.17398034238325136|             434150|      2329.8681884141424|
# |    1|         False|    False|0.11906244021520303|              28172|      2543.3742723271334|
# |    5|         False|    False|0.20957944385018762|            1898419|      1770.1831834805698|
# |    6|         False|    False|0.22410328359932596|            5624838|       1897.910508533757|
# |    7|         False|    False| 0.2237985636963453|            4708466|      1945.5838321865338|


# df3.coalesce(1).write.format("csv").mode('overwrite').save("/user/s2532409/priceTime_Mile")
