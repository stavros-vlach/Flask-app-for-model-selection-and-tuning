from flask import Flask, render_template, request
import pandas as pd
from model import train_model
from utils import preprocess_data
from sklearn.datasets import load_iris

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        model_name = request.form.get("model_name")

        file = request.files.get("dataset")
        if file and file.filename != "":
            df = pd.read_csv(file, sep=None, engine="python")
            X, y, messages = preprocess_data(df)

        else:
            iris = load_iris(as_frame=True)
            df = iris.frame
            df["target"] = iris.target 

        results = train_model(model_name, X, y)
        results["messages"] = messages
            
        return render_template("result.html", results=results)

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
