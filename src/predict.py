from pathlib import Path
import joblib
import numpy as np
from transformers import pipeline
from preprocess import clean_text, remove_sources

ROOT = Path(__file__).resolve().parent.parent      # project folder
MODELS = ROOT / "models"

# Load once at startup (slow), reuse for every prediction (fast)
TFIDF = joblib.load(MODELS / "tfidf.joblib")
LR = joblib.load(MODELS / "lr_model.joblib")
WORDS = TFIDF.get_feature_names_out()
BERT = pipeline("text-classification", model=str(MODELS / "distilbert"),
                truncation=True, max_length=256)

def predict_classical(text):
    x = TFIDF.transform([clean_text(text)])
    p_fake = LR.predict_proba(x)[0][1]
    contrib = x.toarray()[0] * LR.coef_[0]          # + pushes fake, - pushes real
    idx = np.nonzero(contrib)[0]
    order = idx[np.argsort(contrib[idx])]
    return {
        "label": "FAKE" if p_fake >= 0.5 else "REAL",
        "confidence": float(round(max(p_fake, 1 - p_fake) * 100, 1)),
        "fake_words": [WORDS[i] for i in order[::-1][:5] if contrib[i] > 0],
        "real_words": [WORDS[i] for i in order[:5] if contrib[i] < 0],
    }

def predict_bert(text):
    out = BERT(remove_sources(text))[0]
    return {"label": out["label"], "confidence": float(round(out["score"] * 100, 1))}

def predict(text):
    if not text or not clean_text(text):
        raise ValueError("Please enter some text with real words.")
    return {"classical": predict_classical(text), "bert": predict_bert(text)}

if __name__ == "__main__":
    print(predict("BREAKING: Hillary caught hiding SHOCKING secret, watch the video!"))
