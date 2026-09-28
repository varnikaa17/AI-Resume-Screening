import pandas as pd

df = pd.read_csv("data/resume_job_dataset.csv")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nFirst 3 records:")
print(df.head(3).to_string())