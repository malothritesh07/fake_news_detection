import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

wel = pd.read_csv("data/processed/welfake.csv")
liar = pd.read_csv("data/processed/liar_binary.csv")
NAMES = {0: "Real", 1: "Fake"}
COLORS = {0: "#2E7D32", 1: "#C62828"}
FIG = "reports/figures/"

# 1.6 Class balance
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, df) in zip(axes, [("WELFake", wel), ("LIAR", liar)]):
    counts = df["label"].value_counts().reindex([0, 1])
    ax.bar([NAMES[0], NAMES[1]], counts.values, color=[COLORS[0], COLORS[1]])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set_title(f"{name} class balance")
    ax.set_ylabel("Rows")
plt.tight_layout(); plt.savefig(FIG + "class_balance.png", dpi=150); plt.close()

# 1.7 Text length
wel["words"] = wel["content"].str.split().str.len()
plt.figure(figsize=(8, 4))
for lab in [0, 1]:
    plt.hist(wel.loc[wel.label == lab, "words"].clip(upper=2000), bins=60,
             alpha=0.6, label=NAMES[lab], color=COLORS[lab])
plt.xlabel("Words per article (capped at 2,000)"); plt.ylabel("Articles")
plt.title("WELFake text length"); plt.legend()
plt.tight_layout(); plt.savefig(FIG + "text_length.png", dpi=150); plt.close()
print("Median words per article:\n", wel.groupby("label")["words"].median())

# 1.8 Top 20 words per class (5,000-article sample per class, for speed)
sample = wel.groupby("label").sample(5000, random_state=42)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, lab in zip(axes, [0, 1]):
    cv = CountVectorizer(stop_words="english", max_features=20)
    X = cv.fit_transform(sample.loc[sample.label == lab, "content"])
    freq = pd.Series(X.sum(axis=0).A1, index=cv.get_feature_names_out()).sort_values()
    ax.barh(freq.index, freq.values, color=COLORS[lab])
    ax.set_title(f"Top 20 words: {NAMES[lab]}")
    print(f"\nTop words {NAMES[lab]}:", list(freq.index[::-1][:10]))
plt.tight_layout(); plt.savefig(FIG + "top_words.png", dpi=150); plt.close()

# 1.9 Word clouds
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, lab in zip(axes, [0, 1]):
    text = " ".join(sample.loc[sample.label == lab, "content"])
    wc = WordCloud(width=800, height=500, background_color="white",
                   max_words=100).generate(text)
    ax.imshow(wc); ax.axis("off"); ax.set_title(NAMES[lab], fontsize=16)
plt.tight_layout(); plt.savefig(FIG + "wordclouds.png", dpi=150); plt.close()

print("\nSaved 4 charts to reports/figures/")