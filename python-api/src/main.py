import pandas as pd
from pathlib import Path
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# -----------------------------
# Load dataset
# -----------------------------
logging.getLogger(__name__)
dataset = Path("./data/data_cleaned.csv")
realsamples = Path("./data/real_samples_cleaned.csv")

try:
    if dataset.exists() and realsamples.exists():
        df = pd.read_csv(dataset)
        realdata = pd.read_csv(realsamples)
    else:
        raise FileNotFoundError("file not found")
except FileNotFoundError as e:
    logging.error(f"{e} : {dataset} or {realsamples}")

# Features and target
y_data = df["Churn"]
X_data = df.drop(columns=["Churn"])

# Train/test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_data, y_data, test_size=0.3, random_state=10
)

# Second: split the 30% into 15% validation and 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# -----------------------------
# Utility: Evaluation function
# -----------------------------
def evaluate_model(name, model, X_train, y_train,X_val,y_val ,X_test, y_test, proba=False):
    """Train, predict, and print metrics for a given model."""
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    print(f"\n{name} - Training Performance:")
    print(classification_report(y_train, y_pred_train))

    print(f"\n{name} - Validation Performance:")
    print(classification_report(y_val, y_pred_val))
    print("Confusion Matrix (Validation):")
    print(confusion_matrix(y_val, y_pred_val))

    print(f"{name} - Test Performance:")
    print(classification_report(y_test, y_pred_test))
    print("Confusion Matrix (Test):")
    print(confusion_matrix(y_test, y_pred_test))

    if proba:
        y_prob = model.predict_proba(X_test)[:, 1]
        print(f"ROC AUC (Test): {roc_auc_score(y_test, y_prob):.3f}")

# -----------------------------
# Models
# -----------------------------
log_reg = LogisticRegression(max_iter=5000, solver="saga",class_weight="balanced")
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
rf = RandomForestClassifier(
    n_estimators=300, max_depth=10,
    min_samples_split=20, min_samples_leaf=10,
    random_state=42, n_jobs=-1
)

# I got voting classifier as the best model to use here 
# However the recall is the bottleneck
# so i should 
voting_clf = VotingClassifier(
    estimators=[("lr", log_reg), ("rf", rf), ("gb", gb)],
    voting="soft"
)

# -----------------------------
# Run evaluations
# -----------------------------
evaluate_model("Logistic Regression", log_reg, X_train, y_train,X_val,y_val, X_test, y_test, proba=True)
evaluate_model("Gradient Boosting", gb, X_train, y_train,X_val,y_val, X_test, y_test, proba=True)
evaluate_model("Random Forest", rf, X_train, y_train,X_val,y_val, X_test, y_test, proba=True)
evaluate_model("Voting Classifier", voting_clf, X_train, y_train,X_val,y_val, X_test, y_test, proba=True)

# # -----------------------------
# # Correlation heatmap
# # -----------------------------
# plt.figure(figsize=(12, 6))
# sns.heatmap(X_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Feature Correlation Heatmap")
# plt.show()


# -----------------------
#  Testing real data 
# -----------------------
X_new = realdata.drop(columns=["Churn"])

predictions = voting_clf.predict(X_new)
real_samp_pred = log_reg.predict(X_new)
gb_samp_pred = gb.predict(X_new)

actual = realdata["Churn"]
print("real samples predictions:",gb_samp_pred)
print("real churn :",actual)


models = [
    {"name":"logistic_regression.pkl","model":log_reg},
    {"name":"gradient_boosting.pkl","model":gb},
    {"name":"voting_classifier.pkl","model":voting_clf},
    {"name":"random_forest.pkl","model":rf}]

for mod in models:    
    joblib.dump(mod["model"],"./models/" + mod["name"])
