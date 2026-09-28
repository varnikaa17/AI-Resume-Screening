import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

train_text = (
    train_df["resume_text"] + " " +
    train_df["job_description"]
)

test_text = (
    test_df["resume_text"] + " " +
    test_df["job_description"]
)

print("===== DATASET LEAKAGE CHECK =====")

train_duplicates = train_text.duplicated().sum()
test_duplicates = test_text.duplicated().sum()

print(f"Duplicate training samples: {train_duplicates}")
print(f"Duplicate testing samples : {test_duplicates}")

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2
)

X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

similarity = cosine_similarity(X_test, X_train)

max_similarity = similarity.max(axis=1)

print(f"\nAverage max train-test similarity: {max_similarity.mean():.4f}")
print(f"Maximum train-test similarity    : {max_similarity.max():.4f}")

thresholds = [0.90, 0.95, 0.99]

for threshold in thresholds:
    count = (max_similarity >= threshold).sum()
    percentage = count / len(test_df) * 100

    print(
        f"Test samples with similarity >= {threshold}: "
        f"{count} ({percentage:.2f}%)"
    )

print("\nLeakage check completed.")