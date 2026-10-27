# Project Walkthrough: From CSV to Model Comparison

This follows the execution order of the training pipeline.

## 1. Load and validate data

Check that the CSV exists, the Target column is present and non-missing, predictors are numeric, the target contains two classes, and predictors contain no missing values. Fail early when assumptions are not met.

## 2. Separate X and y

X contains candidate predictor columns. y contains Target. Keeping the target out of X prevents target leakage.

## 3. Split before fitting preprocessing

Use train_test_split with test_size=0.20, stratify=y, and random_state=42. This reserves 20% for evaluation, preserves class proportions, and makes the split reproducible.
Reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

## 4. Build model pipelines

Compare Logistic Regression, K-Nearest Neighbors, Random Forest, Gradient Boosting, and a soft Voting Classifier. Each candidate is placed in a pipeline with StandardScaler. The scaler learns from training data and only transforms the held-out samples during evaluation. This avoids fitting preprocessing on test data.
Reference: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

## 5. Fit and evaluate

Fit each candidate on the training partition, predict on the test partition, and calculate accuracy, precision, recall, F1, and ROC AUC where probability estimates are available. Store classification reports and confusion matrices as well.
Reference: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

## 6. Save artifacts

The pipeline is intended to produce a metric table, comparison chart, per-class classification reports, confusion matrices, run metadata, and a serialized selected model. Regenerate these whenever code, data, split, or dependency versions change.

## 7. Test

Run: python -m unittest discover -s tests -v

GitHub Actions is intended to run the tests and validate notebook JSON structure. JSON validation is not the same as executing every notebook cell.

## Target-label limitation

The assignment describes Target as a health outcome/risk variable but does not map numeric values to healthy or unhealthy. Until confirmed, metrics are about numeric label 1 only and cannot be represented as clinical-risk detection.
