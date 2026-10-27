import pandas as pd

df = pd.read_csv("data/processed/welfake.csv")

for label in [0, 1]:
    print(f"\n===== LABEL {label} =====")
    for text in df[df["label"] == label]["content"].sample(5, random_state=1):
        print("-", text[:200], "\n")

# Real news is mostly Reuters, so count Reuters mentions per label
reuters = df["content"].str.contains("Reuters", case=False)
print("Share of rows mentioning Reuters, by label:")
print(reuters.groupby(df["label"]).mean().round(2))
