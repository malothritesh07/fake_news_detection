from predict import predict

tests = {
    "Real news": "The Federal Reserve kept interest rates unchanged on Wednesday, "
                 "saying inflation had eased but remained above its 2 percent target.",
    "Fake/clickbait": "SHOCKING: Obama secretly signed a deal to hand Texas to Mexico, "
                      "share before this video gets deleted!",
    "Short claim": "Says the unemployment rate has doubled in the last two years.",
    "True but casual": "Guys you won't believe it, water literally boils at 100 degrees "
                       "Celsius at sea level, crazy right?!",
    "Empty": "   ",
}

for name, text in tests.items():
    print(f"\n=== {name} ===")
    try:
        r = predict(text)
        c, b = r["classical"], r["bert"]
        print(f"Classical : {c['label']} ({c['confidence']}%)  fake words: {c['fake_words']}  real words: {c['real_words']}")
        print(f"DistilBERT: {b['label']} ({b['confidence']}%)")
    except ValueError as e:
        print("Error shown to user:", e)
        