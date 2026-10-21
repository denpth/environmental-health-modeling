"""
Comprehensive Evaluation Metrics for EPA Air Quality Health Risk Prediction
Author: AI Optimization for Computer Scientists - D682
Description: Implements RMSE, MAPE, and additional metrics for model assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation class for regression tasks.

    Provides multiple evaluation metrics appropriate for health risk prediction:
    - RMSE (Root Mean Square Error)
    - MAPE (Mean Absolute Percentage Error)
    - MAE (Mean Absolute Error)
    - R² (Coefficient of Determination)
    - Additional statistical measures
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.metrics = {}
        self.predictions = None
        self.actual = None

    def calculate_rmse(self, y_true, y_pred):
        """
        Calculate Root Mean Square Error.

        RMSE measures the standard deviation of prediction errors.
        Lower values indicate better model performance.

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            float: RMSE value
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse

    def calculate_mape(self, y_true, y_pred):
        """
        Calculate Mean Absolute Percentage Error.

        MAPE expresses accuracy as a percentage of the error.
        Lower values indicate better model performance.

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            float: MAPE value as percentage
        """
        # Handle division by zero
        mask = y_true != 0
        if mask.sum() == 0:
            return float('inf')

        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape

    def calculate_mae(self, y_true, y_pred):
        """
        Calculate Mean Absolute Error.

        MAE measures the average magnitude of errors in predictions.

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            float: MAE value
        """
        mae = mean_absolute_error(y_true, y_pred)
        return mae

    def calculate_r2(self, y_true, y_pred):
        """
        Calculate R-squared (Coefficient of Determination).

        R² indicates the proportion of variance in the dependent variable
        that is predictable from the independent variables.

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            float: R² value
        """
        r2 = r2_score(y_true, y_pred)
        return r2

    def calculate_additional_metrics(self, y_true, y_pred):
        """
        Calculate additional evaluation metrics.

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            dict: Dictionary of additional metrics
        """
        additional_metrics = {}

        # Explained Variance Score
        additional_metrics['explained_variance'] = explained_variance_score(y_true, y_pred)

        # Mean Bias Error
        additional_metrics['mean_bias_error'] = np.mean(y_pred - y_true)

        # Normalized RMSE (NRMSE)
        rmse = self.calculate_rmse(y_true, y_pred)
        y_range = np.max(y_true) - np.min(y_true)
        additional_metrics['nrmse'] = rmse / y_range if y_range != 0 else float('inf')

        # Pearson Correlation Coefficient
        correlation, p_value = stats.pearsonr(y_true, y_pred)
        additional_metrics['correlation'] = correlation
        additional_metrics['correlation_p_value'] = p_value

        # Residual Statistics
        residuals = y_true - y_pred
        additional_metrics['residual_mean'] = np.mean(residuals)
        additional_metrics['residual_std'] = np.std(residuals)
        additional_metrics['residual_skewness'] = stats.skew(residuals)
        additional_metrics['residual_kurtosis'] = stats.kurtosis(residuals)

        return additional_metrics

    def comprehensive_evaluation(self, y_true, y_pred, model_name="Model"):
        """
        Perform comprehensive evaluation of model predictions.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name (str): Name of the model being evaluated

        Returns:
            dict: Complete evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE MODEL EVALUATION: {model_name}")
        print(f"{'='*60}")

        # Store predictions for later use
        self.actual = y_true
        self.predictions = y_pred

        # Calculate primary metrics
        self.metrics = {
            'rmse': self.calculate_rmse(y_true, y_pred),
            'mape': self.calculate_mape(y_true, y_pred),
            'mae': self.calculate_mae(y_true, y_pred),
            'r2': self.calculate_r2(y_true, y_pred)
        }

        # Calculate additional metrics
        additional = self.calculate_additional_metrics(y_true, y_pred)
        self.metrics.update(additional)

        # Display results
        print(f"\nPrimary Metrics:")
        print(f"  RMSE: {self.metrics['rmse']:.4f}")
        print(f"  MAPE: {self.metrics['mape']:.2f}%")
        print(f"  MAE:  {self.metrics['mae']:.4f}")
        print(f"  R²:   {self.metrics['r2']:.4f}")

        print(f"\nAdditional Metrics:")
        print(f"  Explained Variance: {self.metrics['explained_variance']:.4f}")
        print(f"  Mean Bias Error: {self.metrics['mean_bias_error']:.4f}")
        print(f"  Normalized RMSE: {self.metrics['nrmse']:.4f}")
        print(f"  Correlation: {self.metrics['correlation']:.4f}")

        print(f"\nResidual Analysis:")
        print(f"  Residual Mean: {self.metrics['residual_mean']:.4f}")
        print(f"  Residual Std: {self.metrics['residual_std']:.4f}")
        print(f"  Residual Skewness: {self.metrics['residual_skewness']:.4f}")
        print(f"  Residual Kurtosis: {self.metrics['residual_kurtosis']:.4f}")

        return self.metrics

    def plot_evaluation_results(self, figsize=(15, 10)):
        """
        Create comprehensive visualization of evaluation results.

        Args:
            figsize (tuple): Figure size for the plots
        """
        if self.actual is None or self.predictions is None:
            raise ValueError("Must run comprehensive_evaluation() first.")

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Comprehensive Model Evaluation Results', fontsize=16, fontweight='bold')

        # 1. Actual vs Predicted
        axes[0, 0].scatter(self.actual, self.predictions, alpha=0.6, color='blue')
        axes[0, 0].plot([self.actual.min(), self.actual.max()],
                       [self.actual.min(), self.actual.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)

        # Add R² annotation
        axes[0, 0].text(0.05, 0.95, f'R² = {self.metrics["r2"]:.3f}',
                       transform=axes[0, 0].transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # 2. Residuals vs Predicted
        residuals = self.actual - self.predictions
        axes[0, 1].scatter(self.predictions, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Residuals Distribution
        axes[0, 2].hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_xlabel('Residuals')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Residuals Distribution')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Q-Q Plot for Residuals
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Residuals)')
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Error Distribution
        errors = np.abs(self.actual - self.predictions)
        axes[1, 1].hist(errors, bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Absolute Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Metrics Summary
        axes[1, 2].axis('off')
        metrics_text = f"""
        Primary Metrics:
        RMSE: {self.metrics['rmse']:.4f}
        MAPE: {self.metrics['mape']:.2f}%
        MAE: {self.metrics['mae']:.4f}
        R²: {self.metrics['r2']:.4f}

        Additional Metrics:
        Correlation: {self.metrics['correlation']:.4f}
        NRMSE: {self.metrics['nrmse']:.4f}
        Mean Bias: {self.metrics['mean_bias_error']:.4f}
        """
        axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        axes[1, 2].set_title('Metrics Summary')

        plt.tight_layout()
        plt.savefig('comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_evaluation_report(self, model_name="Model", save_to_file=True):
        """
        Generate a detailed evaluation report.

        Args:
            model_name (str): Name of the model
            save_to_file (bool): Whether to save report to file

        Returns:
            str: Formatted evaluation report
        """
        if not self.metrics:
            raise ValueError("Must run comprehensive_evaluation() first.")

        report = f"""
{'='*80}
DETAILED EVALUATION REPORT: {model_name}
{'='*80}

DATASET SUMMARY:
- Total Samples: {len(self.actual)}
- Target Range: {self.actual.min():.3f} to {self.actual.max():.3f}
- Target Mean: {self.actual.mean():.3f}
- Target Std: {self.actual.std():.3f}

PRIMARY EVALUATION METRICS:
- Root Mean Square Error (RMSE): {self.metrics['rmse']:.4f}
  * Interpretation: Average prediction error magnitude
  * Lower values indicate better performance

- Mean Absolute Percentage Error (MAPE): {self.metrics['mape']:.2f}%
  * Interpretation: Average percentage error
  * Values < 10% = Excellent, 10-20% = Good, 20-50% = Reasonable, >50% = Poor

- Mean Absolute Error (MAE): {self.metrics['mae']:.4f}
  * Interpretation: Average absolute prediction error
  * Lower values indicate better performance

- R-squared (R²): {self.metrics['r2']:.4f}
  * Interpretation: Proportion of variance explained by the model
  * Values closer to 1.0 indicate better performance

ADDITIONAL METRICS:
- Explained Variance Score: {self.metrics['explained_variance']:.4f}
- Normalized RMSE: {self.metrics['nrmse']:.4f}
- Pearson Correlation: {self.metrics['correlation']:.4f}
- Mean Bias Error: {self.metrics['mean_bias_error']:.4f}

RESIDUAL ANALYSIS:
- Residual Mean: {self.metrics['residual_mean']:.4f}
  * Should be close to 0 for unbiased predictions
- Residual Standard Deviation: {self.metrics['residual_std']:.4f}
- Residual Skewness: {self.metrics['residual_skewness']:.4f}
  * Should be close to 0 for normally distributed residuals
- Residual Kurtosis: {self.metrics['residual_kurtosis']:.4f}
  * Should be close to 0 for normally distributed residuals

MODEL PERFORMANCE ASSESSMENT:
"""

        # Performance assessment based on metrics
        if self.metrics['r2'] >= 0.9:
            performance = "Excellent"
        elif self.metrics['r2'] >= 0.8:
            performance = "Very Good"
        elif self.metrics['r2'] >= 0.7:
            performance = "Good"
        elif self.metrics['r2'] >= 0.6:
            performance = "Fair"
        else:
            performance = "Poor"

        report += f"Overall Performance: {performance} (R² = {self.metrics['r2']:.3f})\n"

        if self.metrics['mape'] < 10:
            mape_assessment = "Excellent"
        elif self.metrics['mape'] < 20:
            mape_assessment = "Good"
        elif self.metrics['mape'] < 50:
            mape_assessment = "Reasonable"
        else:
            mape_assessment = "Poor"

        report += f"Prediction Accuracy: {mape_assessment} (MAPE = {self.metrics['mape']:.1f}%)\n"

        report += f"\n{'='*80}\n"

        if save_to_file:
            with open(f'{model_name.lower().replace(" ", "_")}_evaluation_report.txt', 'w') as f:
                f.write(report)
            print(f"Evaluation report saved to {model_name.lower().replace(' ', '_')}_evaluation_report.txt")

        return report

def main():
    """
    Demonstration of the evaluation metrics module.
    """
    print("EPA Air Quality Model Evaluation Metrics")
    print("=" * 50)

    # This would typically be called with actual model predictions
    # For demonstration, we'll create some sample data
    np.random.seed(42)
    n_samples = 200

    # Simulate actual health risk scores
    actual_scores = np.random.normal(10, 1.5, n_samples)
    actual_scores = np.clip(actual_scores, 8, 12)  # Clip to reasonable range

    # Simulate predicted scores with some error
    noise = np.random.normal(0, 0.3, n_samples)
    predicted_scores = actual_scores + noise

    print(f"Demonstrating evaluation with {n_samples} simulated predictions...")

    # Initialize evaluator
    evaluator = ModelEvaluator()

    # Perform comprehensive evaluation
    metrics = evaluator.comprehensive_evaluation(actual_scores, predicted_scores, "Random Forest Demo")

    # Generate visualizations
    evaluator.plot_evaluation_results()

    # Generate detailed report
    report = evaluator.generate_evaluation_report("Random Forest Demo", save_to_file=True)

    print("\nEvaluation complete! Check the generated files:")
    print("- comprehensive_evaluation.png")
    print("- random_forest_demo_evaluation_report.txt")

    return evaluator, metrics

if __name__ == "__main__":
    evaluator, metrics = main()