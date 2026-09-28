import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

resume_train = vectorizer.transform(train_df["resume_text"])
job_train = vectorizer.transform(train_df["job_description"])

resume_test = vectorizer.transform(test_df["resume_text"])
job_test = vectorizer.transform(test_df["job_description"])

train_similarity = []

for i in range(len(train_df)):
    score = cosine_similarity(
        resume_train[i],
        job_train[i]
    )[0][0]
    train_similarity.append(score)

test_similarity = []

for i in range(len(test_df)):
    score = cosine_similarity(
        resume_test[i],
        job_test[i]
    )[0][0]
    test_similarity.append(score)

train_df["tfidf_similarity"] = train_similarity
test_df["tfidf_similarity"] = test_similarity

train_df.to_csv("data/train_features.csv", index=False)
test_df.to_csv("data/test_features.csv", index=False)

print("TF-IDF similarity calculated!")

print("\nTraining similarity:")
print(train_df["tfidf_similarity"].describe())

print("\nTesting similarity:")
print(test_df["tfidf_similarity"].describe())