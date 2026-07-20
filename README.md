# 🩺 Disease Prediction from Medical Data

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

A Machine Learning web app that predicts whether a tumor is **malignant** or **benign** based on patient medical measurements — built as part of the CodeAlpha Machine Learning Internship.

## 📌 Project Overview

This project compares two Machine Learning models — **Logistic Regression** and **Random Forest** — trained on the Breast Cancer Wisconsin dataset, and automatically selects the best-performing one. The final model is deployed through an interactive **Streamlit** web app.

## ✨ Features

- Compares Logistic Regression vs Random Forest and auto-selects the best model
- Interactive Streamlit UI with 5 pages: Home, About Dataset, Model Performance, Disease Prediction, About Project
- Live prediction tool with adjustable input sliders
- Model evaluation: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, Classification Report

## 📊 Dataset

- **Source:** `load_breast_cancer()` — built into scikit-learn
- **Samples:** 569
- **Features:** 30 numeric measurements from cell nuclei images
- **Target:** Malignant or Benign

## 🤖 Algorithms

| Model | Accuracy |
|---|---|
| Logistic Regression | 0.9561 |
| Random Forest | **0.9649** ✅ (Best) |

## 📁 Folder Structure
Disease-Prediction/
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
├── model/
│ └── disease_model.pkl
├── notebooks/
│ └── exploration.ipynb
├── images/
└── src/
├── preprocessing.py
├── evaluation.py
└── utils.py
## ⚙️ Installation

```bash
git clone https://github.com/SameerInnovates/Disease-Prediction.git
cd Disease-Prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 📸 Screenshots

_(Add screenshots here — see the screenshot checklist)_

## ▶️ How to Run

```bash
python train_model.py     # Train and save the model
streamlit run app.py      # Launch the web app
```

## 📈 Results

The Random Forest model achieved **96.49% accuracy** on the test set, outperforming Logistic Regression (95.61%).

## 🚀 Future Improvements

- Add more datasets (e.g., diabetes, heart disease)
- Add cross-validation and hyperparameter tuning
- Add SHAP-based feature importance explanations
- Deploy to Streamlit Community Cloud

## 👤 Author

**[Sameer]**  
CodeAlpha Machine Learning Internship

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.