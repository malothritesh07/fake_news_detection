import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP = set(stopwords.words("english"))
LEMMA = WordNetLemmatizer()

# Source names that give away the label (step 1.11)
SOURCE_RE = re.compile(
    r"\(reuters\)\s*-?|\breuters\b|\bbreitbart\b|21st century wire|\bnew york times\b",
    re.IGNORECASE)

def remove_sources(text):
    return SOURCE_RE.sub(" ", str(text))

def clean_text(text):
    text = remove_sources(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"<.*?>", " ", text)                   # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                # keep letters only
    words = [LEMMA.lemmatize(w) for w in text.split()
             if w not in STOP and len(w) > 2]
    return " ".join(words)

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/processed/welfake.csv")
    has_src = lambda s: s.str.contains(SOURCE_RE)
    before = has_src(df["content"]).groupby(df["label"]).mean().round(3)
    after = has_src(df["content"].map(remove_sources)).groupby(df["label"]).mean().round(3)
    print("Share with source names (0 = real, 1 = fake)")
    print(pd.DataFrame({"before": before, "after": after}))