"""To train the 5 required classifiers and regenerate model artifacts/test_data.csv"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).parent
(ROOT / "model").mkdir(exist_ok=True)
(ROOT / "artifacts").mkdir(exist_ok=True)
dataset = load_breast_cancer(as_frame=True)
X, y = dataset.data, dataset.target
full = X.copy()
full["target"] = y
full.to_csv(ROOT / "artifacts" / "full_dataset_reference.csv", index=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
test = X_test.copy()
test["target"] = y_test
test.to_csv(ROOT / "test_data.csv", index=False)
models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=3000, random_state=42))]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([("scaler", StandardScaler()), ("classifier", SVC(probability=True, random_state=42))]),
}
results, registry = [], {}
for name, estimator in models.items():
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)
    prob = estimator.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name, "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, prob),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred)
    })
    filename = name.lower().replace(" ", "_").replace("-", "") + ".joblib"
    joblib.dump(estimator, ROOT / "model" / filename)
    registry[name] = f"model/{filename}"
pd.DataFrame(results).to_csv(ROOT / "artifacts" / "model_comparison.csv", index=False)
(ROOT / "model" / "model_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
print(pd.DataFrame(results).to_string(index=False))
