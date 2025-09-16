from pyspark import SparkFiles
from pyspark.python.pyspark.shell import sc, spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnull


url = "https://huggingface.co/datasets/Amod/mental_health_counseling_conversations/resolve/main/combined_dataset.json"
spark.sparkContext.addFile(url)  # Spark downloads it for you
local_path = SparkFiles.get("combined_dataset.json")
df = spark.read.option("multiline", "true").json(local_path)
df.createOrReplaceTempView('test')
spark.sql('Select * from test where Response is  null').show()

numbers = list(range(15))
rdd = sc.parallelize(numbers)
print(rdd.count())
print(rdd.max())
print(rdd.mean())
a = rdd.filter(lambda x: x % 3 == 0)
print(a.values())
print(a.collect())
