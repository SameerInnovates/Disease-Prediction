"""
train_model.py
Trains two ML models (Logistic Regression and Random Forest) on the
breast cancer dataset, compares them, and saves the better one.
"""

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data():
    """Load the built-in breast cancer dataset."""
    data = load_breast_cancer()
    X = data.data          # patient measurements (features)
    y = data.target        # 0 = malignant, 1 = benign
    return X, y, data


def train_and_compare(X, y):
    """Train both models and return the one with higher accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model 1: Logistic Regression
    log_model = LogisticRegression(max_iter=5000)
    log_model.fit(X_train, y_train)
    log_preds = log_model.predict(X_test)
    log_acc = accuracy_score(y_test, log_preds)

    # Model 2: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)

    print(f"Logistic Regression Accuracy: {log_acc:.4f}")
    print(f"Random Forest Accuracy:       {rf_acc:.4f}")

    # Pick the better model
    if rf_acc >= log_acc:
        print("Random Forest performed better (or equal). Saving Random Forest model.")
        return rf_model, "Random Forest", rf_acc
    else:
        print("Logistic Regression performed better. Saving Logistic Regression model.")
        return log_model, "Logistic Regression", log_acc


def save_model(model, path="model/disease_model.pkl"):
    """Save the trained model to a file using joblib."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    X, y, data = load_data()
    best_model, best_name, best_acc = train_and_compare(X, y)
    save_model(best_model)
    print(f"\nBest model: {best_name} with accuracy {best_acc:.4f}")