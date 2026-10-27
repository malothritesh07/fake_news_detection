import pandas as pd

RAW = "data/raw/WELFake_Dataset.csv"
OUT = "data/processed/welfake.csv"

def load_welfake():
    df = pd.read_csv(RAW)
    print("Raw rows:", len(df))
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df["title"] = df["title"].fillna("")
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    df["content"] = (df["title"] + " " + df["text"]).str.strip()
    df = df.drop_duplicates(subset=["content"])
    df = df[["content", "label"]]          # label: 0 = real, 1 = fake (checked in 1.4)
    print("Clean rows:", len(df))
    print(df["label"].value_counts())
    df.to_csv(OUT, index=False)
    return df

LIAR_COLS = ["id", "label6", "statement", "subject", "speaker", "job", "state",
             "party", "barely_true", "false", "half_true", "mostly_true",
             "pants_fire", "context"]
FAKE = ["pants-fire", "false", "barely-true"]

def load_liar():
    parts = [pd.read_csv(f"data/raw/liar/{s}.tsv", sep="\t", header=None,
                         names=LIAR_COLS, quoting=3)
             for s in ["train", "valid", "test"]]
    df = pd.concat(parts, ignore_index=True)
    print("\nLIAR rows:", len(df))
    print(df["label6"].value_counts())
    df["label"] = df["label6"].isin(FAKE).astype(int)   # 1 = fake, 0 = real
    df = df.rename(columns={"statement": "content"})[["content", "label"]]
    df = df.dropna().drop_duplicates(subset=["content"])
    print("LIAR binary rows:", len(df))
    print(df["label"].value_counts())
    df.to_csv("data/processed/liar_binary.csv", index=False)
    return df

if __name__ == "__main__":
    load_welfake()
    load_liar()