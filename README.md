# NovaGen Health Risk Classification

A reproducible supervised machine-learning project based on the **NovaGen Research Labs** assignment. It compares five classifiers on a dataset of anonymized-style participant health indicators and reports class-level evaluation metrics.

## Dataset

The included CSV contains **9,549 rows and 23 columns**: 22 input features and `Target`. This differs from the assignment brief's stated 9,800 records, so the actual file's dimensions are reported here rather than assumed from the brief. The file currently has no missing cells or fully duplicated rows.

Features cover:
- Physiological measures: `Age`, `BMI`, `Blood_Pressure`, `Cholesterol`, `Glucose_Level`, `Heart_Rate`
- Lifestyle indicators: `Sleep_Hours`, `Exercise_Hours`, `Water_Intake`, `Stress_Level`, `Smoking`, `Alcohol`, `Diet`
- Health/history indicators: `MentalHealth`, `PhysicalActivity`, `MedicalHistory`, `Allergies`
- Encoded categories: `Diet_Type__Vegan`, `Diet_Type__Vegetarian`, `Blood_Group_AB`, `Blood_Group_B`, `Blood_Group_O`

The CSV contains values `0`, `1`, and `2` in several coded fields including `Smoking` and `Alcohol`, despite the assignment description presenting these as binary. The project intentionally retains the supplied values instead of silently recoding them; validate the dataset's coding dictionary before production use.

## Models compared

1. Logistic Regression (L2 regularization)
2. K-Nearest Neighbors (k=5, Euclidean distance)
3. Random Forest (200 trees)
4. Gradient Boosting (150 estimators)
5. Soft Voting Classifier (Logistic Regression + KNN + Random Forest)

Each model is evaluated on the same stratified 80/20 train-test split using `random_state=42`. `StandardScaler` is fitted within a pipeline using the training partition only. Reported metrics include accuracy, precision, recall and F1 for **target label 1**, plus ROC-AUC where available. Reports and confusion matrices are saved in `reports/`.

### Baseline result from the supplied notebook

The original notebook's saved output reports:
- Logistic Regression — accuracy 81.4%, recall for label 1: 82.8%
- KNN — accuracy 88.3%, recall for label 1: 88.4%
- Random Forest — accuracy 93.8%, recall for label 1: 95.9%
- Gradient Boosting — accuracy 93.0%, recall for label 1: 95.0%
- Voting Classifier — accuracy 91.6%, recall for label 1: 93.1%

These figures are from the notebook's recorded outputs and are not a guarantee of future performance. The runnable pipeline can produce small differences depending on versions and preprocessing. Because the assignment does not define what target label `1` means, “recall for label 1” is used instead of claiming it specifically measures detection of unhealthy/high-risk individuals.

## Repository layout

```text
novagen-health-risk-classification/
├── data/
│   └── novagen_dataset.csv
├── notebooks/
│   ├── novagen_original.ipynb
│   └── analysis_and_model_comparison.ipynb
├── docs/
│   ├── PROJECT_WALKTHROUGH.md       # execution flow, step by step
│   └── SCIKIT_LEARN_CONCEPTS.md     # concepts + official docs links
├── reports/                         # generated evaluation outputs
├── src/
│   ├── __init__.py
│   └── train.py
├── tests/
│   ├── __init__.py
│   └── test_training.py
├── .github/workflows/tests.yml      # tests on push and pull request
├── .gitignore
├── CITATION.cff
├── LICENSE
├── requirements.txt
└── README.md
```

**Original work is preserved:** `notebooks/novagen_original.ipynb` is a copy of the uploaded notebook, retained without rewriting its cells. The enhanced, reproducible training pipeline is in `src/train.py`, and the analysis notebook is separate.

## Run locally

Recommended: Python 3.10 or 3.11.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Train and compare models
python -m src.train

# Optional: specify file locations
python -m src.train --data-path data/novagen_dataset.csv --output-dir reports

# 4. Run tests
python -m unittest discover -s tests -v
```

The training command creates:
- `reports/model_comparison.csv`
- `reports/model_comparison.png`
- `reports/classification_reports.json`
- `reports/confusion_matrices.json`
- `reports/run_metadata.json`
- `reports/best_model.joblib`

The summary CSV plus compact JSON reports are committed as baseline result snapshots. The comparison chart and fitted `.joblib` model are generated locally when you run the pipeline; the binary artifacts are not required to reproduce the project.

## Understand every step

The documentation is organized as a learning path, not just a usage guide:

- [`docs/PROJECT_WALKTHROUGH.md`](docs/PROJECT_WALKTHROUGH.md) explains the pipeline from reading the CSV through validation, splitting, preprocessing, fitting, evaluation, and saving artifacts.
- [`docs/SCIKIT_LEARN_CONCEPTS.md`](docs/SCIKIT_LEARN_CONCEPTS.md) explains the scikit-learn concepts and every model family used, with links to the official scikit-learn documentation.

The explanations are paired with the actual code in `src/train.py`. They use the dataset's numeric target labels rather than guessing which label means healthy or unhealthy.

## Key implementation choices

- **Stratified split:** keeps class proportions similar in train and test partitions.
- **Pipeline-based scaling:** fits preprocessing on the training data only, reducing leakage risk.
- **Multiple metrics:** accuracy alone can hide class-specific errors; precision, recall, F1 and ROC-AUC offer additional perspectives.
- **Reproducibility:** fixed random seeds for the data split and supported estimators.
- **Validation checks:** verifies that the file, target, target classes, and numerical features are present and usable.

## Next improvements

- Confirm the target label meanings and obtain a data dictionary for encoded feature values.
- Add cross-validation and hyperparameter tuning without touching the original notebook.
- Inspect false positives/false negatives and calibrate probabilities if justified.
- Evaluate subgroup performance, potential bias, and dataset shift.
- Add model-card documentation and independent validation before any real-world consideration.

## Official scikit-learn references

The project guide links directly to official scikit-learn resources for [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html), [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), [pipelines](https://scikit-learn.org/stable/modules/compose.html), [classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics), and [ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html). See [`docs/SCIKIT_LEARN_CONCEPTS.md`](docs/SCIKIT_LEARN_CONCEPTS.md) for concept-by-concept explanations and model-specific references.

## Source files

- Assignment brief: `Supervised ML _Assignment5.pdf` (provided separately in the working materials)
- Original notebook: `notebooks/novagen_original.ipynb`
- Dataset: `data/novagen_dataset.csv`