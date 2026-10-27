import tempfile
import unittest
from pathlib import Path

from src.train import TARGET_COLUMN, build_models, evaluate_models, load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "novagen_dataset.csv"


class NovaGenProjectTests(unittest.TestCase):
    def test_dataset_loads_with_expected_shape(self):
        data = load_dataset(DATA_PATH)
        self.assertEqual(data.shape, (9549, 23))
        self.assertIn(TARGET_COLUMN, data.columns)
        self.assertEqual(data[TARGET_COLUMN].nunique(), 2)
        self.assertFalse(data.isna().any().any())

    def test_features_do_not_contain_target(self):
        data = load_dataset(DATA_PATH)
        self.assertEqual(data.drop(columns=[TARGET_COLUMN]).shape[1], 22)
        self.assertNotIn(TARGET_COLUMN, data.drop(columns=[TARGET_COLUMN]).columns)

    def test_all_five_models_are_registered(self):
        self.assertEqual(set(build_models()), {"Logistic Regression", "K-Nearest Neighbors", "Random Forest", "Gradient Boosting", "Soft Voting Classifier"})

    def test_training_produces_metrics(self):
        data = load_dataset(DATA_PATH).sample(n=1200, random_state=42)
        with tempfile.TemporaryDirectory() as temp_dir:
            scores, fitted = evaluate_models(data, temp_dir)
            self.assertEqual(len(scores), 5)
            self.assertEqual(len(fitted), 5)
            self.assertTrue(scores["accuracy"].between(0, 1).all())
            self.assertTrue(scores["recall_class_1"].between(0, 1).all())
            self.assertTrue((Path(temp_dir) / "model_comparison.csv").exists())
            self.assertTrue((Path(temp_dir) / "best_model.joblib").exists())


if __name__ == "__main__":
    unittest.main()
