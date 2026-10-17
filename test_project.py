"""Run after training: python -m unittest -v"""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import train

ROOT = Path(__file__).resolve().parent


class ProjectTests(unittest.TestCase):
    def test_label_mapping(self):
        X, y = train.load_data("classification")
        self.assertEqual(len(X), 569)
        self.assertEqual(int(y.sum()), 212)
        self.assertEqual(set(y.unique()), {0, 1})
        with self.assertRaises(ValueError):
            train.load_data("unknown")

    def test_splits_disjoint_and_complete(self):
        for task in ("regression", "classification"):
            X, y = train.load_data(task)
            a, b, c, d = train.split_data(X, y, task)
            self.assertFalse(set(a.index) & set(b.index))
            self.assertEqual(set(a.index) | set(b.index), set(X.index))
            self.assertEqual(len(c) + len(d), len(y))
            self.assertFalse(X.isna().any().any())

    def test_imputation_and_scaling_training_only(self):
        X, y = train.load_data("classification")
        a, b, c, _ = train.split_data(X, y, "classification")
        a, b = a.copy(), b.copy()
        a.iloc[0, 0] = np.nan
        b.iloc[0, 1] = np.nan
        pipeline = train.make_pipeline(LogisticRegression(max_iter=5000), True).fit(a, c)
        self.assertAlmostEqual(pipeline.named_steps["imputer"].statistics_[0], a.iloc[:, 0].median())
        expected = pipeline.named_steps["imputer"].transform(a).mean(axis=0)
        np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_, expected)
        probability = train.positive_probability(pipeline, b)
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())

    def test_saved_metrics_and_cv_selection(self):
        results = json.loads((ROOT / "results/results.json").read_text())
        for task in ("regression", "classification"):
            rows = results[task]["models"]
            for name, row in rows.items():
                file = ROOT / "results" / task / (name.lower().replace(" ", "_") + "_predictions.csv")
                predictions = pd.read_csv(file)
                self.assertEqual(len(predictions), results[task]["audit"]["test_rows"])
                self.assertEqual(set(predictions.row_id), set(results[task]["audit"]["test_indices"]))
                if task == "regression":
                    measured = train.regression_metrics(predictions.actual, predictions.predicted)
                else:
                    measured = train.classification_metrics(predictions.actual, predictions.predicted,
                                                           predictions.probability_malignant)
                for metric, value in measured.items():
                    self.assertAlmostEqual(value, row["test"][metric], places=10)
            candidates = [name for name in rows if not name.startswith("Dummy")]
            selector = min if task == "regression" else max
            self.assertEqual(selector(candidates, key=lambda name: rows[name]["cv_score_mean"]),
                             results[task]["cv_selected_model"])

    def test_metric_edge_cases(self):
        y = np.array([0, 1, 1, 0])
        result = train.classification_metrics(y, np.zeros(4), np.zeros(4))
        self.assertEqual(result["recall_malignant"], 0)
        self.assertEqual(result["precision_malignant"], 0)
        self.assertEqual(result["fn"], 2)
        self.assertEqual(result["roc_auc"], 0.5)
        self.assertEqual(train.regression_metrics([1, 2, 3], [1, 2, 3])["rmse"], 0)


if __name__ == "__main__":
    unittest.main()
