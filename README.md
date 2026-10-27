# Fake News Detection System

Classifies news text as **REAL** or **FAKE** using a TF-IDF + classical ML baseline and a fine-tuned DistilBERT, served through a Flask web app.

Built by Maloth Ritesh Kumar, Dept. of CSE, NIT Delhi.

## Results

| Model | WELFake F1 | LIAR accuracy | Speed (CPU) |
| --- | --- | --- | --- |
| DistilBERT (fine-tuned) | **0.9912** | 0.519 | 182 ms |
| Linear SVM | 0.9575 | 0.481 | 0.8 ms |
| Logistic Regression | 0.9541 | 0.481 | — |
| Random Forest | 0.9093 | 0.444 | — |
| Multinomial NB | 0.8724 | 0.549 | — |

**Key finding:** every model scores 96–99% on WELFake but falls to about 50% on LIAR (short political claims), below the 55.9% you'd get by always guessing "real". The models learn the *writing style* of WELFake's sources, not truthfulness.

## What I did

- **Data:** WELFake (72,134 → 63,052 articles after removing duplicates and empty rows) and LIAR (12,810 statements, 6 labels → binary).
- **Label check:** found WELFake's real labels are 0 = real, 1 = fake, the opposite of its documentation.
- **Leak fix:** source names (Reuters, NYT, Breitbart) appeared in 89% of real vs 11% of fake articles; removed them before training.
- **Models:** TF-IDF (1–2 grams, 50k terms) with LR, NB, Linear SVM, RF; DistilBERT fine-tuned 2 epochs on a Colab T4 (12 min).
- **Evaluation:** WELFake 80/20 stratified split (seed 42) + cross-dataset test on LIAR.
- **App:** Flask page showing both verdicts, confidence, and the words that pushed the classical model's decision.

## Limitations

- Judges style, not facts: a true statement written casually can be flagged as fake.
- Fails on short claims (LIAR).
- English only.
- DistilBERT is overconfident (often ~100% even when wrong).

## Run it

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4
```

1. Download [WELFake](https://zenodo.org/record/4561253) to `data/raw/` and [LIAR](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) to `data/raw/liar/`.
2. `python src/load_data.py` → `python src/make_split.py` → `python src/train_baseline.py`
3. Fine-tune DistilBERT on Colab and put the model in `models/distilbert/` (not in the repo, ~250 MB).
4. `python app/app.py` → open http://127.0.0.1:5000

## Project structure

```
src/        data loading, preprocessing, training, evaluation, prediction
app/        Flask app + HTML template
reports/    EDA charts, confusion matrices, results CSVs
```

## Sources

- WELFake: Verma et al., [Zenodo](https://zenodo.org/record/4561253)
- LIAR: Wang (2017), ["Liar, Liar Pants on Fire"](https://arxiv.org/abs/1705.00648