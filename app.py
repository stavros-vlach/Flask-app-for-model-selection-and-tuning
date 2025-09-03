from flask import Flask, render_template, request
import pandas as pd
from model import train_model

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        model_name = request.form.get("model_name")

        file = request.files.get("dataset")
        if file:
            df = pd.read_csv(file, sep=None, engine='python')

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            results = train_model(model_name, X, y)
        return render_template("result.html", results=results)

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
