# Environmental Health Risk Modeling

## Project context

Developed during Western Governors University coursework. This repository is a record of my implementation and learning, not a client engagement, deployed operational system, or claim of a new algorithm. Supplied requirements, starter components, datasets, and third-party libraries are distinct from my own implementation. Where their exact provenance or redistribution terms have not been established, they are not represented as my original work or as cleared for public release.


An end-to-end machine learning project that models environmental health risk from weather and air-quality features, then systematically improves a baseline model through feature selection, hyperparameter tuning, regularization, and ensemble learning.

## What it demonstrates

- Exploratory data analysis and feature engineering
- Regression model evaluation using RMSE and R²
- Train/test separation and cross-validation
- Feature selection with `SelectKBest`
- Hyperparameter optimization with `RandomizedSearchCV`
- Regularization with Ridge regression and constrained tree parameters
- Ensemble modeling with Random Forest, Gradient Boosting, XGBoost, and voting ensembles
- Model interpretation and statistical analysis with SHAP and supporting visualizations

## Stack

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, SHAP, SciPy, statsmodels, and Matplotlib.

## Project structure

- `main.py` — primary modeling and evaluation pipeline
- `dataset.csv` — local authorized input, excluded from the public repository
- `requirements.txt` — Python dependencies
- `Working.ipynb` / `working2.ipynb` — exploratory and analysis notebooks
- `figures/` — generated statistical and model-analysis artifacts

## Running the project

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with:

```powershell
.venv\Scripts\activate
```

Then run:

```bash
python main.py dataset.csv --output evaluation.json
```

## Modeling workflow

The pipeline begins with a Random Forest baseline and evaluates holdout and cross-validation performance. It then selects the most informative features, tunes model hyperparameters, applies regularization, trains multiple complementary estimators, and compares optimized performance against the baseline.

The repository also includes exploratory notebooks and generated figures used to examine relationships among environmental variables, feature importance, and predicted health risk.

## Public data and historical results

The original course-supplied scenario, submission documents, spreadsheet, CSV data, and data dictionary have been removed from the current public tree. Notebook outputs were cleared to avoid republishing raw-data previews. Earlier commits may still contain these files; this cleanup does not erase Git history or certify licensing of every retained artifact.

For an independently generated synthetic example, run `python make_demo_data.py` before `python main.py dataset.csv --output evaluation.json`. The generator refuses to overwrite an existing dataset. Alternatively set `MODELING_DATASET` to an authorized CSV with the expected schema. Notebook examples still expect a local `dataset.csv`. The synthetic data is a software fixture, not real health observations, and does not reproduce the historical RMSE/R² reported on the portfolio.

The original experiment is preserved as `historical_main.py` and the two exploratory notebooks. Its feature selection/scaling precede some CV calls; those historical estimates are not fold-contained. The default `main.py` now runs the corrected evaluation in `evaluate.py`.

## Corrected evaluation (October 2026)

All learned imputation, feature selection, and scaling are inside the estimator pipeline. Inner randomized search selects models using training folds; outer cross-validation evaluates that complete selection procedure. A final untouched row holdout is evaluated after tuning. Baseline, per-fold RMSE, variability, R², configuration, dataset hash, and package versions are exported as JSON. No test-set scores guide tuning.

A diagnostic comparison reports prefit versus fold-contained preprocessing under fixed selected model settings. It is labeled diagnostic because settings were selected using the training partition; only the nested estimate includes model selection. New results on the independent synthetic fixture are software-evaluation evidence, not a replacement for historical real-data results or a validated health model. Random splits do not establish future-period or geographic generalization.

Quick software demonstration: `python make_demo_data.py`, then `python main.py dataset.csv --folds 3 --trials 3 --trees 40 --output evaluation.json`. Minimal evaluation dependencies: numpy, pandas, scikit-learn. Historical analyses need the broader requirements file. Run regression checks with `python -m unittest discover -s tests`.
