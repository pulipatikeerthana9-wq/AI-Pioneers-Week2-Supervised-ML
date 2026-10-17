"""Week 2: reproducible supervised learning benchmarks (no network needed).
Run: python train.py --output-dir results
Positive classification label: malignant=1, benign=0.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                    train_test_split)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEED = 42


def load_data(task):
    """Raw diabetes predictors avoid global scaling before the split."""
    if task == "regression":
        data = load_diabetes(as_frame=True, scaled=False)
        return data.data, data.target
    if task == "classification":
        data = load_breast_cancer(as_frame=True)
        return data.data, (data.target == 0).astype(int).rename("malignant")
    raise ValueError("task must be regression or classification")


def split_data(X, y, task):
    return train_test_split(X, y, test_size=0.2, random_state=SEED,
                            stratify=y if task == "classification" else None)


def make_pipeline(model, scale=False):
    # fit() learns preprocessing only from the supplied training fold.
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps + [("model", model)])


def model_specs(task):
    """Small, declared grids; n_jobs=1 is suitable for ordinary laptops."""
    if task == "regression":
        return {
            "Dummy mean": (DummyRegressor(strategy="mean"), False, {}),
            "Linear Regression": (LinearRegression(), True, {}),
            "Decision Tree": (DecisionTreeRegressor(random_state=SEED), False,
                              {"model__max_depth": [3, 5, None],
                               "model__min_samples_leaf": [1, 5, 10]}),
            "Random Forest": (RandomForestRegressor(n_estimators=150,
                                                     random_state=SEED, n_jobs=1), False,
                              {"model__max_depth": [5, None],
                               "model__min_samples_leaf": [1, 5]}),
            "KNN": (KNeighborsRegressor(), True,
                    {"model__n_neighbors": [5, 11, 21],
                     "model__weights": ["uniform", "distance"]}),
        }
    return {
        "Dummy majority": (DummyClassifier(strategy="most_frequent"), False, {}),
        "Logistic Regression": (LogisticRegression(max_iter=5000, random_state=SEED), True,
                                {"model__C": [0.01, 0.1, 1.0, 10.0]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=SEED), False,
                          {"model__max_depth": [3, 5, None],
                           "model__min_samples_leaf": [1, 5]}),
        "Random Forest": (RandomForestClassifier(n_estimators=150,
                                                 random_state=SEED, n_jobs=1), False,
                          {"model__max_depth": [5, None],
                           "model__min_samples_leaf": [1, 3]}),
        "KNN": (KNeighborsClassifier(), True,
                {"model__n_neighbors": [3, 5, 11],
                 "model__weights": ["uniform", "distance"]}),
    }


def positive_probability(model, X):
    classes = model.named_steps["model"].classes_
    positive_column = int(np.flatnonzero(classes == 1)[0])
    return model.predict_proba(X)[:, positive_column]


def regression_metrics(y, prediction):
    return {"mae": float(mean_absolute_error(y, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "r2": float(r2_score(y, prediction))}


def classification_metrics(y, prediction, probability):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, prediction)),
            "precision_malignant": float(precision_score(y, prediction, zero_division=0)),
            "recall_malignant": float(recall_score(y, prediction, zero_division=0)),
            "f1_malignant": float(f1_score(y, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def dataset_hash(X, y):
    values = pd.concat([X, y.rename("target")], axis=1)
    return hashlib.sha256(pd.util.hash_pandas_object(values, index=True).values.tobytes()).hexdigest()


def run_task(task, output):
    output.mkdir(parents=True, exist_ok=True)
    X, y = load_data(task)
    X_train, X_test, y_train, y_test = split_data(X, y, task)
    # Diagnostics concern training data; labels and test distributions do not tune models.
    audit = {"rows": len(X), "features": X.shape[1],
             "train_rows": len(X_train), "test_rows": len(X_test),
             "missing_predictor_values": int(X.isna().sum().sum()),
             "duplicate_predictor_rows": int(X.duplicated().sum()),
             "dataset_sha256": dataset_hash(X, y),
             "train_indices": X_train.index.tolist(), "test_indices": X_test.index.tolist()}
    if task == "classification":
        audit["train_class_counts"] = {str(k): int(v) for k, v in y_train.value_counts().items()}
        audit["test_class_counts"] = {str(k): int(v) for k, v in y_test.value_counts().items()}
    X_train.describe().to_csv(output / "training_summary.csv", index=True)
    pd.concat([X, y.rename("target")], axis=1).to_csv(output / "dataset.csv", index_label="row_id")
    pd.Series(X_train.corr().abs().where(np.triu(np.ones((X.shape[1], X.shape[1])), k=1).astype(bool)).stack()).sort_values(ascending=False).head(5).to_csv(output / "top_training_correlations.csv")
    cv = (KFold(5, shuffle=True, random_state=SEED) if task == "regression"
          else StratifiedKFold(5, shuffle=True, random_state=SEED))
    scoring = "neg_root_mean_squared_error" if task == "regression" else "roc_auc"
    searches = {}
    rows = {}
    # Freeze all selection decisions before inspecting holdout predictions.
    for name, (model, scaled, grid) in model_specs(task).items():
        search = GridSearchCV(make_pipeline(model, scaled), grid, scoring=scoring,
                              cv=cv, n_jobs=1, refit=True, error_score="raise",
                              return_train_score=True)
        search.fit(X_train, y_train)
        searches[name] = search
        best = search.best_index_
        rows[name] = {"best_params": search.best_params_,
                      "cv_score_mean": float(-search.best_score_ if task == "regression" else search.best_score_),
                      "cv_score_std": float(search.cv_results_["std_test_score"][best]),
                      "candidate_count": len(search.cv_results_["params"])}
        safe_name = name.lower().replace(" ", "_")
        pd.DataFrame(search.cv_results_).to_csv(output / (safe_name + "_cv.csv"), index=False)
    candidates = [n for n in rows if not n.startswith("Dummy")]
    winner = (min(candidates, key=lambda n: rows[n]["cv_score_mean"]) if task == "regression"
              else max(candidates, key=lambda n: rows[n]["cv_score_mean"]))
    for name, search in searches.items():
        model = search.best_estimator_
        train_pred, test_pred = model.predict(X_train), model.predict(X_test)
        predictions = pd.DataFrame({"row_id": X_test.index, "actual": y_test.to_numpy(),
                                    "predicted": test_pred})
        if task == "regression":
            train_metrics = regression_metrics(y_train, train_pred)
            test_metrics = regression_metrics(y_test, test_pred)
            predictions["residual_actual_minus_predicted"] = y_test.to_numpy() - test_pred
            rows[name]["worst_errors"] = predictions.assign(absolute_error=lambda d: d["residual_actual_minus_predicted"].abs()).nlargest(3, "absolute_error").to_dict(orient="records")
        else:
            train_probability = positive_probability(model, X_train)
            test_probability = positive_probability(model, X_test)
            train_metrics = classification_metrics(y_train, train_pred, train_probability)
            test_metrics = classification_metrics(y_test, test_pred, test_probability)
            predictions["probability_malignant"] = test_probability
            rows[name]["misclassified_row_ids"] = predictions.loc[predictions.actual != predictions.predicted, "row_id"].tolist()
        rows[name].update({"train": train_metrics, "test": test_metrics})
        predictions.to_csv(output / (name.lower().replace(" ", "_") + "_predictions.csv"), index=False)
        print(task, name, json.dumps(test_metrics), flush=True)
    return {"audit": audit, "selection_metric": "CV RMSE" if task == "regression" else "CV ROC-AUC",
            "cv_selected_model": winner, "models": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {"seed": SEED, "test_fraction": 0.2, "cv_folds": 5,
               "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                               "numpy": np.__version__, "pandas": pd.__version__},
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "command": "python train.py --output-dir results",
               "classification_positive_label": "malignant=1; benign=0"}
    for task in ("regression", "classification"):
        results[task] = run_task(task, args.output_dir / task)
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
    print("CV-selected models:", results["regression"]["cv_selected_model"],
          "/", results["classification"]["cv_selected_model"])


if __name__ == "__main__":
    main()
