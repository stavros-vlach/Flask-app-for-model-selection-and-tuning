from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from utils import plot_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

available_models = {
	"logistic_regression": LogisticRegression,
	"svm": SVC,
	"random_forest": RandomForestClassifier
}


def train_model(model_name, X, y, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_name not in available_models:
        raise ValueError(f"Unknown model: {model_name}")
        
    if model_name in ["logistic_regression", "svm"]:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", available_models[model_name]())
        ])
    else:
        pipeline = Pipeline([
            ("model", available_models[model_name]())
        ])

    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    con_mat = confusion_matrix(y_test, y_pred)
    
    plot_image_str = plot_confusion_matrix(con_mat)
    
    return {
            "model": model_name,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "confusion_matrix": plot_image_str
            }
