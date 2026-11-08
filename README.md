# Machine Learning Assignment 2 — Classification Model Comparison

## a. Problem statement
Build and evaluate five supervised classification algorithms on one public binary classification dataset. The objective is to predict whether a breast tumour is malignant or benign and compare model performance using Accuracy, AUC, Precision, Recall, F1-score and Matthews Correlation Coefficient (MCC). A Streamlit application allows an evaluator to upload labelled test data, select a saved model and inspect metrics and a confusion matrix/classification report.

## b. Dataset description
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC), distributed through `sklearn.datasets.load_breast_cancer` and originally associated with the UCI Machine Learning Repository.
- **Task:** Binary classification.
- **Instances:** 569
- **Predictor features:** 30 numeric features.
- **Target:** `target` (0 = malignant, 1 = benign).
- **Split:** Stratified 80% training / 20% held-out test, `random_state=42`.
- **Test CSV:** `test_data.csv` contains 114 rows, all predictor columns and the target. The app evaluates this data only; it does not retrain models.
- **Preprocessing:** StandardScaler is included inside the Logistic Regression, KNN and SVM pipelines. Decision Tree and Gaussian Naive Bayes are used without feature scaling.

## c. GitHub Repository Link
**Repository:** [PASTE YOUR PUBLIC GITHUB REPOSITORY URL HERE]

Repository contents:
- `app.py` — Streamlit evaluation application.
- `train_models.py` — reproducible training and evaluation script.
- `requirements.txt` — deployment dependencies.
- `test_data.csv` — labelled held-out test data.
- `model/` — five serialized trained models and model registry.
- `artifacts/` — reference dataset, comparison metrics and per-model evaluation summaries.

## d. Models used and evaluation

The same stratified held-out test set was used for all five models. Positive class for metric calculations is class `1` (benign). AUC uses predicted probability for class 1.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| K-Nearest Neighbors | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on model performance

- **Logistic Regression:** Accuracy 0.9825, AUC 0.9954, F1 0.9861, MCC 0.9623. Strong overall discrimination on this held-out split.
- **Decision Tree:** Accuracy 0.9123, AUC 0.9157, F1 0.9286, MCC 0.8174. Good but comparatively lower discrimination on this held-out split.
- **K-Nearest Neighbors:** Accuracy 0.9561, AUC 0.9788, F1 0.9655, MCC 0.9054. Good but comparatively lower discrimination on this held-out split.
- **Gaussian Naive Bayes:** Accuracy 0.9386, AUC 0.9878, F1 0.9517, MCC 0.8676. Strong overall discrimination on this held-out split.
- **Support Vector Machine:** Accuracy 0.9825, AUC 0.9950, F1 0.9861, MCC 0.9623. Strong overall discrimination on this held-out split.

- **Overall winner for this dataset:** Logistic Regression, based on the highest AUC, with F1 and MCC used as tie-breakers. The winner is specific to this dataset split and configuration; it is not universally the best classifier.

## Streamlit application
**Live app:** [PASTE YOUR DEPLOYED STREAMLIT COMMUNITY CLOUD URL HERE]

The app supports CSV upload, classifier selection, six evaluation metrics, a confusion matrix and a classification report. It loads serialized estimators and performs inference only. Upload a CSV with the exact 30 feature columns and a `target` column containing 0/1 labels.

## How to run locally
```bash
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Reproducibility and notes
- Fixed random seed: 42.
- Test set is held out from model fitting.
- Scaling is fitted only on the training partition because it is inside each estimator pipeline.
- Do not use the held-out test set for model fitting.
- Results may vary if package versions, data split, or model hyperparameters are changed.
