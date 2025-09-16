from pyspark import SparkFiles
from pyspark.shell import spark
from pyspark.sql import SparkSession


url = "https://huggingface.co/datasets/Amod/mental_health_counseling_conversations/resolve/main/combined_dataset.json"
url1 = "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW%2Ffinepdfs&config=eng_Latn&split=train&offset=0&length=100"
sp = SparkSession.builder.appName("PySpark").config("spark.some.config.option","some-value").getOrCreate()

taxi = spark.read.csv('/Users/I552581/Library/CloudStorage/OneDrive-SAPSE/Documents/MachineLearning/BaseLine/resources/Taxi_Set.csv')
taxi = spark.read.option('header','true').csv('/Users/I552581/Library/CloudStorage/OneDrive-SAPSE/Documents/MachineLearning/BaseLine/resources/Taxi_Set.csv')
print(taxi)
taxi.printSchema()
print(taxi.count())
tax = taxi.filter(taxi.num_of_passengers > 0)
print(taxi.head(20))
tax.show(2)
