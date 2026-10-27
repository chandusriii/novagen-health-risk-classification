# NovaGen Health Risk Classification

> **Educational machine-learning project — not a medical diagnostic or clinical decision system.**

NovaGen Research Labs' assignment asks for a binary classification model that uses participant health and lifestyle indicators to predict a target health label. This repository provides a reproducible workflow, step-by-step conceptual documentation, model comparison, tests, and the original assignment notebook.

## Project status

This repository is currently being populated. The complete project package is available in the conversation download while the remaining files are transferred.

## Repository goals

- Keep the original assignment notebook unmodified.
- Separate explanatory analysis from the original notebook.
- Explain every major step: validation, train/test splitting, preprocessing, model training, evaluation, and comparison.
- Use scikit-learn's official documentation as the conceptual reference.
- Keep all preprocessing inside model pipelines to avoid fitting preprocessing on the test set.
- Document the target-label ambiguity and avoid clinical claims.

## Intended structure

```text
.
├── .github/workflows/tests.yml
├── data/
│   └── novagen_dataset.csv
├── docs/
│   ├── PROJECT_WALKTHROUGH.md
│   └── SCIKIT_LEARN_CONCEPTS.md
├── notebooks/
│   ├── novagen_original.ipynb
│   ├── analysis_and_model_comparison.ipynb
│   └── original_notebook_note.md
├── reports/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── classification_reports.json
│   ├── confusion_matrices.json
│   └── run_metadata.json
├── src/
│   └── train.py
├── tests/
│   └── test_training.py
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the training pipeline

From the repository root:

```bash
python -m src.train --data-path data/novagen_dataset.csv --output-dir reports
```

The pipeline splits the data using stratification, trains five candidate classifiers, calculates evaluation metrics, and writes CSV/JSON reports and a model-comparison figure. It also saves the selected model as `reports/best_model.joblib` when the pipeline is run.

## Run tests and validate notebooks

```bash
python -m unittest discover -s tests -v
python -c "import nbformat; from pathlib import Path; [nbformat.validate(nbformat.read(p, as_version=4)) for p in Path('notebooks').glob('*.ipynb')]; print('Notebook validation passed')"
```

GitHub Actions runs the unit tests and notebook-schema validation on pushes and pull requests to `main`.

## Workflow, explained

1. **Load and validate data.** Confirm the CSV exists, the target is present and non-missing, predictors are numeric, and the target has two classes.
2. **Separate features and target.** `X` contains predictor columns; `y` contains `Target`.
3. **Create a hold-out set.** Use a stratified train/test split so class proportions are preserved in each partition.
4. **Build pipelines.** Standardize features inside a scikit-learn `Pipeline`; this keeps scaler fitting limited to training data. Tree models generally do not require feature scaling, but using the same explicit structure simplifies comparison.
5. **Train candidate classifiers.** Logistic Regression, K-Nearest Neighbors, Random Forest, Gradient Boosting, and a soft Voting Classifier are compared.
6. **Evaluate.** Accuracy, precision, recall, F1, ROC AUC (where available), classification reports, and confusion matrices are recorded.
7. **Compare models.** Results are sorted by recall for target label `1`; the chosen model is a selection for this metric on this split, not proof of clinical superiority.
8. **Save artifacts.** Metrics, plot, metadata, and the fitted model are written to `reports/`.

See [Project Walkthrough](docs/PROJECT_WALKTHROUGH.md) and [scikit-learn Concepts](docs/SCIKIT_LEARN_CONCEPTS.md) for detail.

## Model comparison

Generated results belong in `reports/model_comparison.csv` and `reports/model_comparison.png`. Model scores can change if the dataset, dependency versions, or split configuration changes. Do not hard-code a model ranking into documentation without regenerating the report from the current data and code.

## Target-label and safety note

The assignment describes `Target` as a health outcome or risk variable but does not clearly define which numeric value means “healthy” or “unhealthy.” Therefore, metrics described as recall for class 1 refer strictly to **numeric target label 1**. Confirm the label mapping with the assignment owner before interpreting a prediction as a real-world health state. This project is for supervised-learning education and has not been clinically validated.

## Official scikit-learn references

- [An introduction to machine learning with scikit-learn](https://scikit-learn.org/stable/tutorial/basic/tutorial.html)
- [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html)
- [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [Cross-validation and model selection](https://scikit-learn.org/stable/modules/cross_validation.html)

## License and citation

See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
