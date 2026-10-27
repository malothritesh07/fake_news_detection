import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

train = pd.read_csv("data/processed/train.csv").dropna(subset=["text_clean"])
test = pd.read_csv("data/processed/test.csv").dropna(subset=["text_clean"])

# 2.5 TF-IDF
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=5, sublinear_tf=True)
X_train = tfidf.fit_transform(train["text_clean"])     # learn vocabulary from train only
X_test = tfidf.transform(test["text_clean"])
y_train, y_test = train["label"], test["label"]
joblib.dump(tfidf, "models/tfidf.joblib")
print("TF-IDF shape:", X_train.shape)

# 2.6 Four models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=4),
    "Multinomial NB": MultinomialNB(alpha=0.1),
    "Linear SVM": LinearSVC(C=0.5),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
}
results = []
for name, m in models.items():
    t = time.time()
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    results.append({"model": name,
                    "accuracy": round(accuracy_score(y_test, pred), 4),
                    "f1": round(f1_score(y_test, pred), 4),
                    "train_sec": round(time.time() - t, 1)})
    print(f"{name}: done in {results[-1]['train_sec']}s")

res = pd.DataFrame(results).sort_values("f1", ascending=False)
print("\n", res.to_string(index=False))
res.to_csv("reports/baseline_results.csv", index=False)

# 2.7 Save best + LR
best = res.iloc[0]["model"]
joblib.dump(models[best], "models/best_model.joblib")
joblib.dump(models["Logistic Regression"], "models/lr_model.joblib")
print("\nBest model:", best)

# 2.8 Sanity check: top words per class from LR
words = tfidf.get_feature_names_out()
coefs = models["Logistic Regression"].coef_[0]
order = coefs.argsort()
print("\nTop REAL words:", list(words[order[:20]]))
print("Top FAKE words:", list(words[order[-20:][::-1]]))