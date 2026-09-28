import pandas as pd
import re

df = pd.read_csv("data/resume_job_dataset.csv")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z0-9+#.\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df["resume_text"] = df["resume_text"].apply(clean_text)
df["job_description"] = df["job_description"].apply(clean_text)

df.to_csv(
    "data/cleaned_resume_dataset.csv",
    index=False
)

print("Preprocessing completed!")
print("Rows:", len(df))
print("\nSample cleaned resume:")
print(df["resume_text"].iloc[0])

print("\nSample cleaned job description:")
print(df["job_description"].iloc[0])