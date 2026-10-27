from transformers import pipeline

clf = pipeline("text-classification", model="models/distilbert",
               truncation=True, max_length=256)

tests = [
    "The Federal Reserve held interest rates steady on Wednesday, citing stable inflation.",
    "BREAKING: Hillary caught hiding SHOCKING secret, watch the video before it gets deleted!",
]
for t in tests:
    print(clf(t)[0], "<-", t[:60])