import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

def find_csv():
    paths = glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True)
    if not paths:
        raise FileNotFoundError(
            "No CSV found. Download the Kaggle Clickbait Dataset and put its CSV file inside data/raw/."
        )
    # Prefer a file containing 'clickbait' in its name.
    paths.sort(key=lambda p: ("clickbait" not in os.path.basename(p).lower(), len(p)))
    return paths[0]

def normalize_columns(df):
    mapping = {}
    for c in df.columns:
        clean = re.sub(r"[^a-z0-9]+", "", str(c).lower())
        if clean in {"headline", "title", "text", "headlines"}:
            mapping[c] = "Headline"
        elif clean in {"clickbait", "label", "isclickbait", "target"}:
            mapping[c] = "Clickbait"
    df = df.rename(columns=mapping)
    if "Headline" not in df.columns or "Clickbait" not in df.columns:
        raise ValueError(
            f"Could not identify Headline and Clickbait columns. Found: {list(df.columns)}"
        )
    return df[["Headline", "Clickbait"]].copy()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)

csv_path = find_csv()
print("Using:", csv_path)

df = pd.read_csv(csv_path)
df = normalize_columns(df)

df["Headline"] = df["Headline"].fillna("").astype(str)
df["Clickbait"] = pd.to_numeric(df["Clickbait"], errors="coerce")
df = df.dropna(subset=["Clickbait"])
df["Clickbait"] = df["Clickbait"].astype(int)
df = df[df["Clickbait"].isin([0, 1])]
df = df.drop_duplicates(subset=["Headline"]).reset_index(drop=True)
df = df[df["Headline"].str.strip().ne("")].reset_index(drop=True)

df["Clean_Headline"] = df["Headline"].apply(clean_text)
df["Word_Count"] = df["Headline"].str.split().str.len()
df["Character_Count"] = df["Headline"].str.len()

out = os.path.join(OUT_DIR, "clickbait_dataset.csv")
df.to_csv(out, index=False)

print("\nDataset shape:", df.shape)
print("\nClass distribution:")
print(df["Clickbait"].value_counts())

# Class distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Clickbait")
plt.xticks([0, 1], ["Not Clickbait", "Clickbait"])
plt.title("Clickbait Class Distribution")
plt.tight_layout()
plt.savefig("reports/class_distribution.png", dpi=200)
plt.close()

# Headline length distribution
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="Word_Count", hue="Clickbait", bins=30, element="step")
plt.title("Headline Word Count Distribution")
plt.tight_layout()
plt.savefig("reports/headline_length_distribution.png", dpi=200)
plt.close()

# Word clouds
for label, name in [(1, "clickbait"), (0, "non_clickbait")]:
    text = " ".join(df.loc[df["Clickbait"] == label, "Clean_Headline"])
    if text.strip():
        wc = WordCloud(width=1000, height=500, background_color="white").generate(text)
        wc.to_file(f"reports/{name}_wordcloud.png")

print("\nSaved processed dataset and EDA reports.")
