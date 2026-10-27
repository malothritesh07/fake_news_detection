import time
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import clean_text, remove_sources

def prepare(df, name):
    t = time.time()
    df["text_bert"] = df["content"].map(remove_sources).str.strip()   # for DistilBERT
    df["text_clean"] = df["content"].map(clean_text)                  # for TF-IDF
    df = df[df["text_clean"].str.len() > 0]                           # drop now-empty rows
    print(f"{name}: {len(df):,} rows cleaned in {time.time() - t:.0f}s")
    return df[["text_bert", "text_clean", "label"]]

wel = prepare(pd.read_csv("data/processed/welfake.csv"), "WELFake")
liar = prepare(pd.read_csv("data/processed/liar_binary.csv"), "LIAR")

train, test = train_test_split(wel, test_size=0.2, stratify=wel["label"], random_state=42)

train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
liar.to_csv("data/processed/liar_clean.csv", index=False)

print(f"\nTrain: {len(train):,} | Test: {len(test):,} | LIAR: {len(liar):,}")
print("Fake share  train:", round(train.label.mean(), 3),
      "| test:", round(test.label.mean(), 3))