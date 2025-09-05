from datasets import load_dataset
import pandas as pd
import boto3

full_ds = load_dataset("Navya1602/Personality_dataset", split="train")
df = pd.DataFrame(full_ds)
full_ds.to_csv("personality_dataset.csv")

bucket_name = "ml-personality-data"  # Replace with your actual S3 bucket
file_path = "personality_dataset.csv"
s3_key = "datasets/personality_dataset.csv"  # How it will appear in S3

# Upload
s3 = boto3.client("s3")
s3.upload_file(file_path, bucket_name, s3_key)

print(f"✅ Uploaded to s3://{bucket_name}/{s3_key}")