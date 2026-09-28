import pandas as pd
import joblib
import matplotlib.pyplot as plt

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report
)

train_df = pd.read_csv("data/train_features.csv")
test_df = pd.read_csv("data/test_features.csv")

train_text = (
    train_df["resume_text"] + " " +
    train_df["job_description"]
)

test_text = (
    test_df["resume_text"] + " " +
    test_df["job_description"]
)

y_train = train_df["label"]
y_test = test_df["label"]

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_text = vectorizer.fit_transform(train_text)
X_test_text = vectorizer.transform(test_text)

similarity_train = train_df["tfidf_similarity"].values.reshape(-1, 1)
similarity_test = test_df["tfidf_similarity"].values.reshape(-1, 1)

X_train = hstack([
    X_train_text,
    similarity_train
])

X_test = hstack([
    X_test_text,
    similarity_test
])

model = LogisticRegression(
    C=2.0,
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)

print("\n===== TF-IDF + SIMILARITY RESULTS =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, predictions))

fpr, tpr, _ = roc_curve(y_test, probabilities)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("TF-IDF + Similarity ROC Curve")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    "results/similarity_model_roc.png",
    dpi=300
)

plt.show()

joblib.dump(
    vectorizer,
    "models/similarity_vectorizer.pkl"
)

joblib.dump(
    model,
    "models/similarity_model.pkl"
)

print("\nModel saved successfully!")
print("ROC curve saved successfully!")