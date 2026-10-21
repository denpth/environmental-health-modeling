# Random Forest pipeline (earlier version)

An earlier, standalone version of the health-risk model from the same WGU course (AI optimization), before the work was combined into the main pipeline in the repository root.

- `data_preprocessing.py`: loads `dataset.csv`, handles missing values, encodes categorical columns, and splits the data
- `random_forest_model.py`: trains and tunes a `RandomForestRegressor` with grid search and cross-validation, then plots feature importance and predictions
- `evaluation_metrics.py`: computes RMSE, MAE and R² plus residual diagnostics, and saves an evaluation chart
- `algorithm_research_report.md`: comparison of the candidate algorithms and why Random Forest was chosen
- `feature_importance.png`, `model_predictions.png`: output from one run

The dataset is not included (see the root README). With an authorized `dataset.csv` in this folder:

```bash
python random_forest_model.py
python evaluation_metrics.py
```

The trained model file (`health_risk_rf_model.joblib`, about 5.7 MB) is not committed. Running the training script recreates it.
