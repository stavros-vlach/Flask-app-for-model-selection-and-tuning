from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from model import train_model
from utils import preprocess_data, predict_with_pipeline
from sklearn.datasets import load_iris
import joblib

app = Flask(__name__)

def safe_cast(val):
    try:
        if val.isdigit():
            return int(val)
        return float(val)
    except ValueError:
        return val 

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    feature_names = []
    if request.method == "POST":
        model_name = request.form.get("model_name")
        
        parameters = {
        key: safe_cast(value)
        for key, value in request.form.items()
        if key != "model_name" and key != "dataset" and value != ""
    }
        file = request.files.get("dataset")
        if file and file.filename != "":
            df = pd.read_csv(file, sep=None, engine="python")
        else:
            iris = load_iris(as_frame=True)
            df = iris.frame
            df["target"] = iris.target 
            
        X, y, cat_encoder, label_encoder, categorical_cols, numeric_cols, feature_names_before, feature_names_after, message = preprocess_data(df)
        results = train_model(model_name, X, y, **parameters)
        
        joblib.dump({"pipeline": results["pipeline"], "feature_names_before": feature_names_before, 
        "feature_names_after": feature_names_after, "one_hot_encoder": cat_encoder,
        "label_encoder": label_encoder, "categorical_cols": categorical_cols, "numeric_cols": numeric_cols},"saved_model.pkl")
        
        return render_template("result.html", results=results, feature_names=feature_names, message=message)
    
    return render_template("index.html", results=results, feature_names=feature_names)
    
@app.route("/predict", methods=["GET", "POST"])
def predict():
    model_dict = joblib.load("saved_model.pkl")
    feature_names = model_dict["feature_names_before"]
    if request.method == "POST":
        input_dict = {col: request.form.get(col) for col in feature_names}
        y_pred = predict_with_pipeline(input_dict)
        return render_template("predict_results.html", prediction=y_pred)
    return render_template("predict.html", feature_names=feature_names)

if __name__ == "__main__":
    app.run(debug=True)
