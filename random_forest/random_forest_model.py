"""
Random Forest Model for EPA Air Quality and Health Risk Prediction
Author: AI Optimization for Computer Scientists - D682
Description: Implements Random Forest algorithm for predicting health risk scores
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our data preprocessing module
from data_preprocessing import AirQualityDataProcessor

class HealthRiskRandomForest:
    """
    Random Forest model specifically designed for health risk prediction based on air quality data.

    This class provides:
    - Hyperparameter tuning using GridSearchCV
    - Model training and validation
    - Feature importance analysis
    - Performance evaluation with multiple metrics
    - Model persistence for deployment
    """

    def __init__(self, random_state=42):
        """
        Initialize the Random Forest model.

        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.training_score = None
        self.validation_scores = None
        self.is_trained = False

    def tune_hyperparameters(self, X_train, y_train, cv_folds=5):
        """
        Perform hyperparameter tuning using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training target
            cv_folds (int): Number of cross-validation folds
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)

        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        print(f"Testing {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])} parameter combinations...")

        # Initialize base model
        rf_base = RandomForestRegressor(random_state=self.random_state)

        # Perform grid search
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Store best parameters
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        print(f"\nBest parameters found:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")

        print(f"Best cross-validation score: {-grid_search.best_score_:.4f} (RMSE)")

        return self.best_params

    def train_model(self, X_train, y_train, tune_hyperparams=True):
        """
        Train the Random Forest model.

        Args:
            X_train: Training features
            y_train: Training target
            tune_hyperparams (bool): Whether to perform hyperparameter tuning
        """
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)

        if tune_hyperparams:
            # Perform hyperparameter tuning
            self.tune_hyperparameters(X_train, y_train)
        else:
            # Use default parameters
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state
            )
            self.model.fit(X_train, y_train)

        # Calculate training score
        train_predictions = self.model.predict(X_train)
        self.training_score = {
            'rmse': np.sqrt(mean_squared_error(y_train, train_predictions)),
            'mae': mean_absolute_error(y_train, train_predictions),
            'r2': r2_score(y_train, train_predictions)
        }

        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=5, scoring='neg_mean_squared_error'
        )
        self.validation_scores = {
            'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
            'cv_rmse_std': np.sqrt(cv_scores.std()),
            'cv_scores': np.sqrt(-cv_scores)
        }

        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.is_trained = True

        print(f"Model training completed!")
        print(f"Training RMSE: {self.training_score['rmse']:.4f}")
        print(f"Training R²: {self.training_score['r2']:.4f}")
        print(f"Cross-validation RMSE: {self.validation_scores['cv_rmse_mean']:.4f} ± {self.validation_scores['cv_rmse_std']:.4f}")

        return self.model

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on test data.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation. Call train_model() first.")

        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate evaluation metrics
        evaluation_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }

        print(f"Test Set Evaluation Results:")
        print(f"  RMSE: {evaluation_metrics['rmse']:.4f}")
        print(f"  MAE:  {evaluation_metrics['mae']:.4f}")
        print(f"  R²:   {evaluation_metrics['r2']:.4f}")
        print(f"  MAPE: {evaluation_metrics['mape']:.2f}%")

        # Store predictions for visualization
        self.test_predictions = y_pred
        self.test_actual = y_test

        return evaluation_metrics

    def plot_feature_importance(self, top_n=15, figsize=(10, 8)):
        """
        Plot feature importance from the trained Random Forest model.

        Args:
            top_n (int): Number of top features to display
            figsize (tuple): Figure size for the plot
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting. Call train_model() first.")

        plt.figure(figsize=figsize)

        # Get top N features
        top_features = self.feature_importance.head(top_n)

        # Create horizontal bar plot
        plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - Random Forest Model')
        plt.gca().invert_yaxis()

        # Add value labels on bars
        for i, v in enumerate(top_features['importance']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center')

        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

        return top_features

    def plot_predictions(self, figsize=(12, 5)):
        """
        Plot actual vs predicted values and residuals.

        Args:
            figsize (tuple): Figure size for the plots
        """
        if not hasattr(self, 'test_predictions'):
            raise ValueError("Model must be evaluated before plotting. Call evaluate_model() first.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Actual vs Predicted plot
        ax1.scatter(self.test_actual, self.test_predictions, alpha=0.6, color='blue')
        ax1.plot([self.test_actual.min(), self.test_actual.max()],
                [self.test_actual.min(), self.test_actual.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Health Risk Score')
        ax1.set_ylabel('Predicted Health Risk Score')
        ax1.set_title('Actual vs Predicted Values')
        ax1.grid(True, alpha=0.3)

        # Residuals plot
        residuals = self.test_actual - self.test_predictions
        ax2.scatter(self.test_predictions, residuals, alpha=0.6, color='green')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Health Risk Score')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, filename='health_risk_rf_model.joblib'):
        """
        Save the trained model to disk.

        Args:
            filename (str): Filename for the saved model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving. Call train_model() first.")

        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'best_params': self.best_params,
            'training_score': self.training_score,
            'validation_scores': self.validation_scores,
            'timestamp': datetime.now().isoformat()
        }

        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")

        return filename

    def load_model(self, filename='health_risk_rf_model.joblib'):
        """
        Load a previously trained model from disk.

        Args:
            filename (str): Filename of the saved model
        """
        model_data = joblib.load(filename)

        self.model = model_data['model']
        self.feature_importance = model_data['feature_importance']
        self.best_params = model_data['best_params']
        self.training_score = model_data['training_score']
        self.validation_scores = model_data['validation_scores']
        self.is_trained = True

        print(f"Model loaded from {filename}")
        print(f"Model trained on: {model_data['timestamp']}")

        return self.model

def main():
    """
    Main execution function demonstrating the complete Random Forest pipeline.
    """
    print("EPA Air Quality Health Risk Prediction - Random Forest Model")
    print("=" * 70)

    try:
        # Step 1: Data Preprocessing
        print("Step 1: Data Preprocessing")
        processor = AirQualityDataProcessor('dataset.csv')
        processor.load_data()
        processor.explore_data()
        processor.clean_data()
        processor.engineer_features()
        processor.prepare_features()
        X_train, X_test, y_train, y_test = processor.split_data()

        # Step 2: Model Training
        print("\nStep 2: Model Training")
        rf_model = HealthRiskRandomForest(random_state=42)
        rf_model.train_model(X_train, y_train, tune_hyperparams=True)

        # Step 3: Model Evaluation
        print("\nStep 3: Model Evaluation")
        evaluation_metrics = rf_model.evaluate_model(X_test, y_test)

        # Step 4: Feature Analysis
        print("\nStep 4: Feature Importance Analysis")
        top_features = rf_model.plot_feature_importance(top_n=15)

        print("\nTop 10 Most Important Features:")
        for i, (_, row) in enumerate(top_features.head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']}: {row['importance']:.4f}")

        # Step 5: Prediction Visualization
        print("\nStep 5: Prediction Visualization")
        rf_model.plot_predictions()

        # Step 6: Save Model
        print("\nStep 6: Model Persistence")
        model_filename = rf_model.save_model()

        # Step 7: Generate Summary Report
        print("\n" + "="*70)
        print("FINAL MODEL SUMMARY")
        print("="*70)

        print(f"Dataset: {processor.df.shape[0]} samples, {len(processor.feature_names)} features")
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        print(f"\nModel Performance:")
        print(f"  Training RMSE: {rf_model.training_score['rmse']:.4f}")
        print(f"  Test RMSE: {evaluation_metrics['rmse']:.4f}")
        print(f"  Test R²: {evaluation_metrics['r2']:.4f}")
        print(f"  Test MAPE: {evaluation_metrics['mape']:.2f}%")

        print(f"\nBest Hyperparameters:")
        for param, value in rf_model.best_params.items():
            print(f"  {param}: {value}")

        print(f"\nModel saved as: {model_filename}")

        return rf_model, processor, evaluation_metrics

    except Exception as e:
        print(f"Error in Random Forest pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    model, processor, metrics = main()