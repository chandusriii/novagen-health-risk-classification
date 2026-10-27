# NovaGen Health Risk Classification

A reproducible, hands-on supervised machine learning project exploring classification algorithms with Python and Scikit-learn.

> **Educational project only:** This is not a clinically validated medical tool. The assignment does not explicitly define the meaning of target labels `0` and `1`; therefore, results are reported by numeric label rather than assuming which one means “healthy” or “unhealthy.”

## Project Overview

The goal of this project is to build a stronger understanding of supervised learning through implementation, experimentation, and model comparison. It covers the end-to-end workflow—from exploring and validating the dataset to preprocessing, training, evaluation, and reporting results.

### Algorithms explored
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Gradient Boosting
- Soft Voting Classifier

### Concepts practiced
- Dataset inspection and validation
- Stratified train/test splitting
- Feature scaling with Scikit-learn pipelines
- Classification metrics: accuracy, precision, recall, F1-score, and ROC-AUC where available
- Confusion matrices and model comparison
- Reproducible training and automated tests

## Baseline Model Results

The original notebook's saved outputs report the following accuracy scores:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 81.4% |
| K-Nearest Neighbors | 88.3% |
| Random Forest | 93.8% |
| Gradient Boosting | 93.0% |
| Soft Voting Classifier | 91.6% |

These are saved baseline results from the supplied notebook, not a guarantee of future performance. For detailed class-level metrics, see the reports in `reports/`.

## Dataset

The included CSV contains **9,549 rows and 23 columns**: 22 input features and the `Target` column. See the notebook and project walkthrough for more information about the dataset and feature groups. Coded values should be interpreted using a verified data dictionary before any real-world use.

## Repository Highlights

- `notebooks/novagen_original.ipynb` — original notebook preserved separately.
- `notebooks/analysis_and_model_comparison.ipynb` — guided exploration and comparison.
- `src/train.py` — reusable model training pipeline.
- `docs/PROJECT_WALKTHROUGH.md` — step-by-step explanation of the workflow.
- `docs/SCIKIT_LEARN_CONCEPTS.md` — concepts and links to official Scikit-learn documentation.
- `tests/` — automated tests for key training behavior.
- `reports/` — saved baseline evaluation outputs and generated report locations.

## Run Locally

Recommended: Python 3.10 or 3.11.

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m src.train
python -m unittest discover -s tests -v
```

The training pipeline saves model-comparison metrics, classification reports, confusion matrices, run metadata, and a fitted model under `reports/`.

## Learning Resources

- [Project walkthrough](docs/PROJECT_WALKTHROUGH.md)
- [Scikit-learn concepts guide](docs/SCIKIT_LEARN_CONCEPTS.md)
- [Training pipeline](src/train.py)

This repository is a learning project focused on understanding how algorithms work, when they may be useful, and how to evaluate their results—not just calling a model's `.fit()` method.
