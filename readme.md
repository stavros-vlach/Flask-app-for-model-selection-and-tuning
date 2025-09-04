# ML Model Trainer Web App
A web application that allows users to train classification models (Logistic Regression, SVM, Random Forest) on any dataset and view evaluation metrics and the confusion matrix.

## Features
- Supports multiple ML models: Logistic Regression, SVM, Random Forest
- Allows selection of hyperparameters for each model
- Data preprocessing:
  - One-hot encoding for categorical features
  - Label encoding for categorical targets
  - Standard scaling for numerical features (for Logistic Regression and SVM)
- Displays evaluation metrics: Accuracy, Recall, Precision, F1-score
- Confusion matrix visualization
- Upload your own dataset (default: Iris dataset)

## Installation & Running Instructions
1. Clone the repository:
https://github.com/stavros-vlach/Flask-app-for-model-selection-and-tuning
2. Create a virtual environment and install dependencies:
python3 -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
pip install -r requirements.txt
3. Run the Flask application:
python3 app.py
4. Open in your browser:
http://127.0.0.1:5000

## Deployment
The application is deployed on Render: [👉 Try the App Here.](https://flask-app-for-model-selection-and-tuning.onrender.com/?fbclid=IwY2xjawMmgCdleHRuA2FlbQIxMQABHmBwHSXHZCKb9X5am5wP6bH_HmWImbQldUL-f5zx-ETLez-3okX-HzRXfXET_aem_dxv08-zYl05m_9e-jjGQ1g)



https://flask-app-for-model-selection-and-tuning.onrender.com/
