import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)

train = pd.read_csv("data/processed/train.csv").dropna(subset=["text_clean"])
test = pd.read_csv("data/processed/test.csv").dropna(subset=["text_clean"])
liar = pd.read_csv("data/processed/liar_clean.csv").dropna(subset=["text_clean"])

tfidf = joblib.load("models/tfidf.joblib")
X_train = tfidf.transform(train["text_clean"])
X_test = tfidf.transform(test["text_clean"])
X_liar = tfidf.transform(liar["text_clean"])

# 3.1 Score 4 classical models on WELFake test + LIAR (same settings as Phase 2)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=4),
    "Multinomial NB": MultinomialNB(alpha=0.1),
    "Linear SVM": LinearSVC(C=0.5),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
}
def score(y, p):
    return (round(accuracy_score(y, p), 4), round(precision_score(y, p), 4),
            round(recall_score(y, p), 4), round(f1_score(y, p), 4))

rows = []
for name, m in models.items():
    m.fit(X_train, train["label"])
    for ds, X, y in [("WELFake", X_test, test["label"]), ("LIAR", X_liar, liar["label"])]:
        acc, prec, rec, f1 = score(y, m.predict(X))
        rows.append({"model": name, "dataset": ds, "accuracy": acc,
                     "precision": prec, "recall": rec, "f1": f1})
    print(name, "scored")

# DistilBERT numbers from the Colab run (step 2.4 / 2.9)
rows += [{"model": "DistilBERT", "dataset": "WELFake", "accuracy": 0.9922, "f1": 0.9912},
         {"model": "DistilBERT", "dataset": "LIAR", "accuracy": 0.5191, "f1": 0.4492}]
res = pd.DataFrame(rows)
res.to_csv("reports/results.csv", index=False)
print("\n", res.pivot(index="model", columns="dataset", values=["accuracy", "f1"]).round(4))

# 3.2 Confusion matrices for Linear SVM
svm = models["Linear SVM"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (ds, X, y) in zip(axes, [("WELFake test", X_test, test["label"]),
                                 ("LIAR", X_liar, liar["label"])]):
    ConfusionMatrixDisplay.from_predictions(y, svm.predict(X), display_labels=["Real", "Fake"],
                                            ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Linear SVM: {ds}")
plt.tight_layout(); plt.savefig("reports/figures/confusion_matrices.png", dpi=150); plt.close()
print("\nSaved confusion_matrices.png")

# 3.3 Speed: ms per article (100 test articles)
sample = test.sample(100, random_state=1)
t = time.time()
svm.predict(tfidf.transform(sample["text_clean"]))
print(f"\nLinear SVM: {(time.time() - t) * 10:.2f} ms per article")

from transformers import pipeline
bert = pipeline("text-classification", model="models/distilbert", truncation=True, max_length=256)
t = time.time()
bert(list(sample["text_bert"]), batch_size=16)
print(f"DistilBERT (CPU): {(time.time() - t) * 10:.1f} ms per article")