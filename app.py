import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).parent
REGISTRY = json.loads((ROOT / "model" / "model_registry.json").read_text(encoding="utf-8"))

st.set_page_config(page_title="Breast Cancer Classifier Comparison", page_icon="🧪", layout="wide")
st.title("Classification Model Evaluation Dashboard")
st.caption("Compare five trained classifiers on an unseen labelled test CSV. Models are loaded from disk; this app does not retrain them.")

st.markdown("""
**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)  
**Task:** Binary classification of breast tumour diagnosis.  
**Target encoding:** `0 = malignant`, `1 = benign`.  
Upload a CSV containing all 30 predictor columns and a `target` column.
""")

uploaded = st.file_uploader("Upload labelled test data (CSV)", type=["csv"])
if uploaded is None:
    st.info("Upload `test_data.csv` from the GitHub repository to evaluate the saved models.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

if "target" not in df.columns:
    st.error("The CSV must include a `target` column.")
    st.stop()

model_names = list(REGISTRY.keys())
selected = st.selectbox("Select classifier", model_names)
model_path = ROOT / REGISTRY[selected]
model = joblib.load(model_path)

feature_names = list(getattr(model, "feature_names_in_", []))
# Pipeline and some estimators may not expose feature_names_in_ consistently; infer from reference dataset.
reference_path = ROOT / "artifacts" / "full_dataset_reference.csv"
reference_columns = pd.read_csv(reference_path, nrows=1).columns.tolist()
required = [c for c in reference_columns if c != "target"]
missing = [c for c in required if c not in df.columns]
extra = [c for c in df.columns if c not in required + ["target"]]
if missing:
    st.error(f"Missing required feature columns ({len(missing)}): {', '.join(missing)}")
    st.stop()
if extra:
    st.warning(f"Ignoring extra columns: {', '.join(extra)}")

X = df[required]
y_true = df["target"]
if y_true.isna().any() or not set(y_true.unique()).issubset({0, 1}):
    st.error("Target must contain only non-null binary values 0 and 1.")
    st.stop()

try:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
except Exception as exc:
    st.error(f"Prediction failed. Check feature values and data types. Details: {exc}")
    st.stop()

auc = roc_auc_score(y_true, probabilities) if y_true.nunique() == 2 else float("nan")
metric_values = {
    "Accuracy": accuracy_score(y_true, predictions),
    "AUC": auc,
    "Precision": precision_score(y_true, predictions, zero_division=0),
    "Recall": recall_score(y_true, predictions, zero_division=0),
    "F1": f1_score(y_true, predictions, zero_division=0),
    "MCC": matthews_corrcoef(y_true, predictions),
}
st.subheader(f"Results: {selected}")
cols = st.columns(6)
for col, (label, value) in zip(cols, metric_values.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("#### Confusion matrix")
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for (i, j), value in __import__("numpy").ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown("#### Classification report")
    report = classification_report(y_true, predictions, target_names=["malignant (0)", "benign (1)"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

st.markdown("#### Test data preview")
st.dataframe(df.head(10), use_container_width=True)
