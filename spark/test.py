from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnull

spark= SparkSession.builder.getOrCreate()
print(spark)


url = "https://huggingface.co/datasets/Amod/mental_health_counseling_conversations/resolve/main/combined_dataset.json"

spark.sparkContext.addFile(url)                      # Spark downloads it for you
local_path = SparkFiles.get("combined_dataset.json")
df = spark.read.option("multiline", "true").json(local_path)
df.show(3, truncate=False)
df.head()
print(df.count())
print(df.schema.names)
print(df['Context'])
df.select("Context").show()
# df.filter("Context!= aaaaa").show()
df.filter("Context !=  'asdsadsadas'").select('Response')
# (df.filter("passenger = 1").select("trip_distance,total_amount")
#  .sort("trip_distance", False).where("non-airport rides = 1"))

print(df.filter(isnull(col("Context"))).count())
df1 = df.fillna({"Context": '1'})
df1.head()
df2 = df1.withColumnRenamed("Context","as")
print(df2.head())
df2.show(1)