# scikit-learn Concept Guide

This guide follows the official scikit-learn documentation and connects each concept to this project.

## Features and target

In supervised learning, features (X) are the input variables and y is the label to predict. Here, Target is the label and all other columns are candidate predictors.

Reference: https://scikit-learn.org/stable/supervised_learning.html

## Train/test split

A hold-out test set estimates performance on observations not used during fitting. train_test_split creates the partitions. stratify=y preserves target class proportions, and random_state makes the split reproducible.

Reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

## Scaling and data leakage

StandardScaler centers numeric features and scales them to unit variance. Scaling can matter for distance-based models such as KNN and optimization-based estimators such as Logistic Regression. A scaler must be fit on training data only; fitting it before the split leaks test-set information. A Pipeline fits preprocessing with the estimator in one operation.

References: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html ; https://scikit-learn.org/stable/modules/compose.html ; https://scikit-learn.org/stable/common_pitfalls.html#data-leakage

## Candidate models

### Logistic Regression
A linear classifier that estimates class probabilities with a logistic function. Despite its name, it is commonly used for classification.
Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

### K-Nearest Neighbors
Predicts from labels of nearby training examples. Feature scaling matters because distances can otherwise be dominated by large-unit features. This project uses k=5 and Euclidean distance.
Reference: https://scikit-learn.org/stable/modules/neighbors.html

### Random Forest
An ensemble of decision trees whose predictions are aggregated. It can model nonlinear relationships and usually does not require feature scaling.
Reference: https://scikit-learn.org/stable/modules/ensemble.html#forest

### Gradient Boosting
Builds an additive model stage by stage, using new trees to improve the loss. Learning rate and tree count/depth influence model complexity.
Reference: https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting

### Soft Voting Classifier
Combines probability estimates from its component models and chooses the class with the largest averaged probability. Probability quality matters for soft voting.
Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html

## Evaluation metrics

- Accuracy: fraction of test examples predicted correctly.
- Precision for label 1: fraction of predictions labeled 1 that are truly label 1.
- Recall for label 1: fraction of true label-1 examples found by the model.
- F1: harmonic mean of precision and recall.
- ROC AUC: measures ranking performance across thresholds in binary classification.
- Confusion matrix: counts true labels against predicted labels.

References: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics ; https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html ; https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

## Model selection

This assignment uses a single stratified hold-out set for a straightforward baseline. A single split can be noisy. For more robust comparisons, use cross-validation and conduct hyperparameter search within training data only, leaving a final test set untouched until choices are complete.

References: https://scikit-learn.org/stable/modules/cross_validation.html ; https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

## Limitation

The assignment does not explicitly map numeric target labels to healthy or unhealthy. Results for label 1 must not be interpreted clinically until this mapping is confirmed. This is an educational project, not a clinically validated model.
