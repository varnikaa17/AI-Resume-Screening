import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

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

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("TF-IDF extraction completed!")
print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)