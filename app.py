"""
app.py
Streamlit web app for the Disease Prediction project.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from src.evaluation import get_metrics, get_confusion_matrix, get_classification_report

# Page config
st.set_page_config(page_title="Disease Prediction", layout="wide")

# Load data and model (cached so it doesn't reload every click)
@st.cache_resource
def load_everything():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = joblib.load("model/disease_model.pkl")
    return data, X_test, y_test, model


data, X_test, y_test, model = load_everything()

# Sidebar navigation
page = st.sidebar.radio(
    "Navigate",
    ["Home", "About Dataset", "Model Performance", "Disease Prediction", "About Project"],
)

# ---------------- HOME ----------------
if page == "Home":
    st.title("🩺 Disease Prediction App")
    st.write(
        "This app predicts whether a tumor is **malignant** or **benign** "
        "based on patient medical measurements, using Machine Learning."
    )
    st.write("Use the sidebar to explore the dataset, check model performance, "
             "or try a live prediction.")

# ---------------- ABOUT DATASET ----------------
elif page == "About Dataset":
    st.title("📊 About the Dataset")
    st.write(
        "This project uses the **Breast Cancer Wisconsin dataset**, built into "
        "scikit-learn. It contains measurements from cell nuclei images."
    )
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    st.write(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1] - 1}")
    st.dataframe(df.head(10))

# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":
    st.title("📈 Model Performance")
    preds = model.predict(X_test)
    metrics = get_metrics(y_test, preds)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['precision']:.4f}")
    col3.metric("Recall", f"{metrics['recall']:.4f}")
    col4.metric("F1 Score", f"{metrics['f1']:.4f}")

    st.subheader("Confusion Matrix")
    cm = get_confusion_matrix(y_test, preds)
    st.write(pd.DataFrame(cm,
                           index=["Actual Malignant", "Actual Benign"],
                           columns=["Predicted Malignant", "Predicted Benign"]))

    st.subheader("Classification Report")
    st.text(get_classification_report(y_test, preds, target_names=data.target_names))

# ---------------- DISEASE PREDICTION ----------------
elif page == "Disease Prediction":
    st.title("🔮 Try a Prediction")
    st.write("Adjust the sliders below to enter patient measurements:")

    input_data = []
    cols = st.columns(3)
    for i, feature in enumerate(data.feature_names):
        col = cols[i % 3]
        min_val = float(data.data[:, i].min())
        max_val = float(data.data[:, i].max())
        mean_val = float(data.data[:, i].mean())
        val = col.slider(feature, min_val, max_val, mean_val)
        input_data.append(val)

    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        prediction_proba = model.predict_proba(input_array)[0]

        result = data.target_names[prediction]
        st.success(f"Prediction: **{result.upper()}**")
        st.write(f"Confidence: {prediction_proba[prediction]:.2%}")

# ---------------- ABOUT PROJECT ----------------
elif page == "About Project":
    st.title("ℹ️ About This Project")
    st.write(
        "This is a Machine Learning internship project (CodeAlpha) that predicts "
        "disease possibility from patient data. It compares Logistic Regression "
        "and Random Forest models and uses the best-performing one for predictions."
    )
    st.write("**Tech stack:** Python, scikit-learn, Streamlit, pandas, numpy, matplotlib")