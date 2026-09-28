import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/cleaned_resume_dataset.csv")

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

train_df.to_csv(
    "data/train.csv",
    index=False
)

test_df.to_csv(
    "data/test.csv",
    index=False
)

print("Dataset split completed!")
print("Total samples:", len(df))
print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

print("\nTraining labels:")
print(train_df["label"].value_counts())

print("\nTesting labels:")
print(test_df["label"].value_counts())