import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def preprocess_data(df):
    messages = []
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    target = df.iloc[:, -1]  
    features = df.iloc[:, :-1]

    if any(col in features.columns for col in categorical_cols):
        features = pd.get_dummies(features, columns=[c for c in categorical_cols if c in features.columns])
        messages.append(f"Categorical columns {categorical_cols} were converted to one-hot vectors.")

    if target.dtype == "object" or str(target.dtype).startswith("category"):
        target = LabelEncoder().fit_transform(target)
        messages.append("The target column was label-encoded into numeric values.")

    return features.values, target, messages

def plot_confusion_matrix(cm, class_names=None):
    if class_names is None:
        class_names = np.arange(cm.shape[0])  
        
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    img_str = base64.b64encode(buf.read()).decode('utf-8')


    return img_str
