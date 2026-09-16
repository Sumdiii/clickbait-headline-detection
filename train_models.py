import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

DATA = "data/processed/clickbait_dataset.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

df = pd.read_csv(DATA)
X = df["Clean_Headline"].fillna("")
y = df["Clickbait"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=250, random_state=42, n_jobs=-1
    ),
    "Linear SVM": LinearSVC(random_state=42)
}

results = []
trained = {}

for name, classifier in models.items():
    print(f"\nTraining {name}...")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ("classifier", classifier)
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained[name] = pipe
    safe_name = name.lower().replace(" ", "_")
    joblib.dump(pipe, f"{MODEL_DIR}/{safe_name}.pkl")

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Clickbait", "Clickbait"],
        yticklabels=["Not Clickbait", "Clickbait"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/confusion_{safe_name}.png", dpi=200)
    plt.close()

    with open(f"{REPORT_DIR}/classification_report_{safe_name}.txt", "w") as f:
        f.write(classification_report(
            y_test, pred,
            target_names=["Not Clickbait", "Clickbait"],
            digits=4
        ))

results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
results_df.to_csv(f"{REPORT_DIR}/model_comparison.csv", index=False)

# Comparison chart
plot_df = results_df.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
    var_name="Metric",
    value_name="Score"
)
plt.figure(figsize=(11, 6))
sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.title("Model Performance Comparison")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/model_comparison.png", dpi=200)
plt.close()

best_name = results_df.iloc[0]["Model"]
joblib.dump(trained[best_name], f"{MODEL_DIR}/best_model.pkl")

with open(f"{MODEL_DIR}/model_info.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "test_size": 0.20,
        "random_state": 42,
        "tfidf_max_features": 5000,
        "ngram_range": [1, 2]
    }, f, indent=2)

print("\n===== RESULTS =====")
print(results_df.to_string(index=False))
print(f"\nBest model: {best_name}")
print("Saved models and reports.")
