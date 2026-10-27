import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flask import Flask, render_template, request
from predict import predict

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    text, result, error = "", None, None
    if request.method == "POST":
        text = request.form.get("text", "")
        try:
            result = predict(text)
        except ValueError as e:
            error = str(e)
    return render_template("index.html", text=text, result=result, error=error)

if __name__ == "__main__":
    app.run(debug=False)   # debug off: avoids loading DistilBERT twice