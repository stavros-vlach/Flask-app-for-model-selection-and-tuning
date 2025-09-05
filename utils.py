import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import joblib

def preprocess_data(df):
    dropped_cols = df.isna().sum()
    message = None
    if np.any(dropped_cols) != 0:
        message = f"Number of NaN in each column: {dropped_cols}"
    df = df.dropna(axis=0, how='any')

    target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names_before = list(X.columns)
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    
    cat_encoder = None
    if categorical_cols:
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat_encoded = cat_encoder.fit_transform(X[categorical_cols])

        new_cols = cat_encoder.get_feature_names_out(categorical_cols)

        X_cat_encoded_df = pd.DataFrame(X_cat_encoded, columns=new_cols, index=X.index)

        X = pd.concat([X[numeric_cols], X_cat_encoded_df], axis=1)
    else:
        categorical_cols = None
        
    if not numeric_cols:
        numeric_cols = None
        
    feature_names_after = list(X.columns)
    
    label_encoder = None
    if y.dtype == "object" or str(y.dtype) == "category":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    return X, y, cat_encoder, label_encoder, categorical_cols, numeric_cols, feature_names_before, feature_names_after, message

def predict_with_pipeline(input_dict):
    model_dict = joblib.load("saved_model.pkl")
    pipeline = model_dict["pipeline"]
    feature_names = model_dict["feature_names_after"]
    categorical_cols = model_dict.get("categorical_cols", [])
    label_encoder = model_dict.get("label_encoder", None)
    cat_encoder = model_dict.get("one_hot_encoder", None)
    numeric_cols = model_dict.get("numeric_cols", [])
    
    X_new_raw = pd.DataFrame([input_dict])
    if numeric_cols and categorical_cols:
        X_new_numeric = X_new_raw[numeric_cols]
        X_new_categorical = X_new_raw[categorical_cols]
        X_new_cat_encoded = cat_encoder.transform(X_new_categorical)
        encoded_cols = cat_encoder.get_feature_names_out(categorical_cols)
        X_new_cat_df = pd.DataFrame(X_new_cat_encoded, columns=encoded_cols, index=X_new_raw.index)
        X_new_processed = pd.concat([X_new_numeric, X_new_cat_df], axis=1)
    elif numeric_cols:
        X_new_processed = X_new_raw[numeric_cols]
    else:
        X_new_categorical = X_new_raw[categorical_cols]
        X_new_cat_encoded = cat_encoder.transform(X_new_categorical)
        encoded_cols = cat_encoder.get_feature_names_out(categorical_cols)
        X_new_processed = pd.DataFrame(X_new_cat_encoded, columns=encoded_cols, index=X_new_raw.index)
        
    X_final = X_new_processed.reindex(columns=feature_names, fill_value=0)
    #print(feature_names)
    #print("-------------------------------------------------------------------------------------------")
    
    y_pred_encoded = pipeline.predict(X_final)[0]
    
    if label_encoder:
        y_pred = label_encoder.inverse_transform([y_pred_encoded])[0]
    else:
        y_pred = y_pred_encoded
        
    return y_pred

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
