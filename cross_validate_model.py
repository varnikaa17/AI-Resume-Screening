
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

df = pd.read_csv("data/resume_job_dataset.csv")

df = df.dropna(subset=["resume_text", "job_description", "label"])
df["label"] = df["label"].astype(int)

y = df["label"].values
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []
f1_scores = []
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(
    folds.split(df, y), start=1
):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    train_text = (
        train_df["resume_text"] + " " +
        train_df["job_description"]
    )
    val_text = (
        val_df["resume_text"] + " " +
        val_df["job_description"]
    )

    X_train_text = vectorizer.fit_transform(train_text)
    X_val_text = vectorizer.transform(val_text)

    train_resume = vectorizer.transform(train_df["resume_text"])
    train_job = vectorizer.transform(train_df["job_description"])
    val_resume = vectorizer.transform(val_df["resume_text"])
    val_job = vectorizer.transform(val_df["job_description"])

    train_sim = np.asarray(
        train_resume.multiply(train_job).sum(axis=1)
    ).reshape(-1, 1)

    val_sim = np.asarray(
        val_resume.multiply(val_job).sum(axis=1)
    ).reshape(-1, 1)

    X_train = hstack([X_train_text, train_sim]).tocsr()
    X_val = hstack([X_val_text, val_sim]).tocsr()

    model = LogisticRegression(
        C=2.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, train_df["label"])
    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(val_df["label"], predictions)
    f1 = f1_score(val_df["label"], predictions)
    auc = roc_auc_score(val_df["label"], probabilities)

    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    auc_scores.append(auc)

    print(f"\nFold {fold}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")

print("\n===== CROSS-VALIDATION SUMMARY =====")
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Mean F1:       {np.mean(f1_scores):.4f}")
print(f"Mean ROC-AUC:  {np.mean(auc_scores):.4f}")
print(f"ROC-AUC Std:   {np.std(auc_scores):.4f}")
