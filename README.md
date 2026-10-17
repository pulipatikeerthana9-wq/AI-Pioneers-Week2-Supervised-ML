# AI Pioneers — Week 2: Supervised Machine Learning

Public-data educational project covering Linear Regression, Logistic Regression, Decision Trees, Random Forest and KNN. Four nonbaseline algorithms per task plus dummy baselines; five-fold training-only CV, fixed 80/20 holdout, leakage-safe pipelines and error analysis.

## Files
- `train.py`: load, preprocess, tune, train, predict and evaluate both tasks.
- `test_project.py`: five automated tests.
- `requirements.txt`: recorded dependency versions.
- `results/results.json`: full measured metrics, parameters, indices and hashes.
- `results/regression/` and `results/classification/`: exported public datasets, training summaries, CV search outputs and test predictions.
- `Week2_Supervised_Machine_Learning_Report.docx`: four-page report.
- `submission_description.txt`: portal description (over 200 words).
- `verification.txt`: recorded test and repeat-run status.

## Windows setup
Install Python 3.13 (the recorded run used 3.13.13). Extract the ZIP, open its folder in a terminal, then run:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --output-dir results
.\.venv\Scripts\python.exe -m unittest -v
```

Using the environment's Python directly avoids PowerShell activation-policy issues. On macOS/Linux, create with `python3 -m venv .venv`, then replace the executable with `.venv/bin/python`.

No dataset download or credentials are needed. Dependencies require internet to install. Training takes a few minutes depending on hardware. A SciPy optimizer deprecation warning may appear in this recorded environment; it did not prevent training or tests. Exact numeric agreement is verified only in the recorded environment, and other versions may produce small differences.

## Findings
CV selected Linear Regression for diabetes (test RMSE 53.85, MAE 42.79, R² 0.453) and Logistic Regression for classification (accuracy 96.49%, malignant recall 0.9286, AUC 0.9960). These are model metrics, NOT internship grades. Classification positive label is malignant=1, benign=0. Random Forest ties classifier accuracy but misses four malignant records rather than three; the report explains this and other trade-offs. Regression is moderately predictive rather than artificially inflated to 90%.

## Before submission
1. Read the code and report, rerun both commands, and understand train/test leakage, metrics and errors.
2. Follow your internship provider's AI-use and disclosure policy; it was not supplied here. Adapt statements to your actual contribution. Do not claim independent authorship or validation you did not perform.
3. Create a GitHub repository named `AI-Pioneers-Week2` in your own account. Upload the extracted project files, not only this ZIP. Use a public repository if your evaluator requires direct access.
4. Upload the Word report to the portal. Paste the contents of `submission_description.txt` into the description field.
5. Copy your repository's actual URL from GitHub into the project field. No Week 2 repository has been created by this package. Do not submit a placeholder URL.
6. If requested by the provider, add your exact student name/ID to the Word document; no identity was invented here.

No score is guaranteed. This is a public-data benchmark, not a medical diagnostic tool. No external submission or GitHub upload has been performed.

## Data documentation
- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- https://scikit-learn.org/stable/common_pitfalls.html
