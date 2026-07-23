# """
# Model evaluation module
# """

# import pandas as pd
# import numpy as np
# import logging
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# class ModelEvaluator:
#     """Handles model evaluation and results saving"""
    
#     def __init__(self, config):
#         self.config = config
#         self.logger = logging.getLogger(__name__)
    
#     def evaluate_all_models(self, trained_models, X, y):
#         """
#         Evaluate all trained models
        
#         Args:
#             trained_models: Dictionary of trained models with data splits
#             X: Full feature matrix (for reference)
#             y: Full target vector (for reference)
            
#         Returns:
#             dict: Evaluation results for all models
#         """
#         self.logger.info("Starting model evaluation...")
        
#         results = {}
        
#         for model_name, model_data in trained_models.items():
#             self.logger.info(f"Evaluating {model_name}...")
            
#             try:
#                 # Get model and test data
#                 model = model_data['model']
#                 X_test = model_data['X_test']
#                 y_test = model_data['y_test']
                
#                 # Make predictions
#                 y_pred = model.predict(X_test)
#                 y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
                
#                 # Calculate metrics
#                 metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
                
#                 # Store results
#                 results[model_name] = {
#                     'metrics': metrics,
#                     'y_true': y_test,
#                     'y_pred': y_pred,
#                     'y_pred_proba': y_pred_proba
#                 }
                
#                 # Log results
#                 self.log_model_results(model_name, metrics)
                
#             except Exception as e:
#                 self.logger.error(f"Failed to evaluate {model_name}: {str(e)}")
#                 continue
        
#         self.logger.info(f"Evaluation completed for {len(results)} models")
#         return results
    
#     def calculate_metrics(self, y_true, y_pred, y_pred_proba):
#         """
#         Calculate all required metrics
        
#         Args:
#             y_true: True labels
#             y_pred: Predicted labels
#             y_pred_proba: Predicted probabilities
            
#         Returns:
#             dict: Dictionary of calculated metrics
#         """
#         metrics = {
#             'accuracy': accuracy_score(y_true, y_pred),
#             'precision': precision_score(y_true, y_pred, average='binary'),
#             'recall': recall_score(y_true, y_pred, average='binary'),
#             'f1_score': f1_score(y_true, y_pred, average='binary'),
#             'roc_auc': roc_auc_score(y_true, y_pred_proba)
#         }
        
#         return metrics
    
#     def log_model_results(self, model_name, metrics):
#         """
#         Log model evaluation results
        
#         Args:
#             model_name: Name of the model
#             metrics: Dictionary of metrics
#         """
#         self.logger.info(f"{model_name} Results:")
#         for metric_name, value in metrics.items():
#             self.logger.info(f"  {metric_name}: {value:.4f}")
    
#     def save_results(self, results):
#         """
#         Save evaluation results to CSV
        
#         Args:
#             results: Dictionary of evaluation results
#         """
#         self.logger.info("Saving evaluation results...")
        
#         # Prepare data for CSV
#         rows = []
#         for model_name, model_results in results.items():
#             row = {'Model': model_name}
#             row.update(model_results['metrics'])
#             rows.append(row)
        
#         # Create DataFrame
#         df_results = pd.DataFrame(rows)
        
#         # Ensure results directory exists
#         self.config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
#         # Save to CSV
#         df_results.to_csv(self.config.METRICS_FILE, index=False)
#         self.logger.info(f"Results saved to {self.config.METRICS_FILE}")
        
#         # Print results summary
#         self.print_results_summary(df_results)
        
#         # Create visualization
#         self.create_results_visualization(df_results)
    
#     def print_results_summary(self, df_results):
#         """
#         Print results summary to console
        
#         Args:
#             df_results: DataFrame with results
#         """
#         print("\n" + "="*80)
#         print("MODEL EVALUATION RESULTS")
#         print("="*80)
        
#         # Format and display results
#         pd.set_option('display.precision', 4)
#         pd.set_option('display.width', None)
#         pd.set_option('display.max_columns', None)
        
#         print(df_results.to_string(index=False))
        
#         # Find best model for each metric
#         print("\n" + "-"*80)
#         print("BEST PERFORMING MODELS:")
#         print("-"*80)
        
#         metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
#         for metric in metrics:
#             if metric in df_results.columns:
#                 best_idx = df_results[metric].idxmax()
#                 best_model = df_results.loc[best_idx, 'Model']
#                 best_score = df_results.loc[best_idx, metric]
#                 print(f"{metric.upper()}: {best_model} ({best_score:.4f})")
    
#     def create_results_visualization(self, df_results):
#         """
#         Create visualization of results
        
#         Args:
#             df_results: DataFrame with results
#         """
#         try:
#             # Ensure plots directory exists
#             self.config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            
#             # Create bar plot of metrics
#             metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            
#             plt.figure(figsize=(12, 8))
            
#             # Prepare data for plotting
#             models = df_results['Model'].values
#             x = np.arange(len(models))
#             width = 0.15
            
#             for i, metric in enumerate(metrics):
#                 if metric in df_results.columns:
#                     values = df_results[metric].values
#                     plt.bar(x + i*width, values, width, label=metric.upper())
            
#             plt.xlabel('Models')
#             plt.ylabel('Score')
#             plt.title('Model Performance Comparison')
#             plt.xticks(x + width*2, models, rotation=45, ha='right')
#             plt.legend()
#             plt.grid(True, alpha=0.3)
#             plt.tight_layout()
            
#             # Save plot
#             plot_path = self.config.PLOTS_DIR / "model_comparison.png"
#             plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#             plt.close()
            
#             self.logger.info(f"Results visualization saved to {plot_path}")
            
#         except Exception as e:
#             self.logger.warning(f"Failed to create visualization: {str(e)}")



"""
evaluate.py

Model evaluation module for DTI Prediction.
Evaluates trained models and saves metrics and visualizations.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.config import RESULTS_DIR, PLOTS_DIR


class ModelEvaluator:
    """Handles model evaluation and result saving"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate_all_models(self, trained_models, X=None, y=None):
        """
        Evaluate all trained models.

        Parameters
        ----------
        trained_models : dict
            Dictionary containing trained models and test data

        Returns
        -------
        list of dict
            Evaluation metrics for each model
        """

        self.logger.info("Starting model evaluation...")
        results = []

        for model_name, data in trained_models.items():
            self.logger.info(f"Evaluating model: {model_name}")

            try:
                model = data["model"]
                X_test = data["X_test"]
                y_test = data["y_test"]

                y_pred = model.predict(X_test)
                y_prob = (
                    model.predict_proba(X_test)[:, 1]
                    if hasattr(model, "predict_proba")
                    else None
                )

                metrics = {
                    "model": model_name,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1_score": f1_score(y_test, y_pred, zero_division=0),
                    "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None
                }

                self._log_metrics(model_name, metrics)
                results.append(metrics)

            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {str(e)}")

        self.logger.info("Model evaluation completed")
        return results

    def save_results(self, results):
        """
        Save evaluation results to CSV and generate plots.
        """

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(results)
        output_path = RESULTS_DIR / "metrics_tuned_rf.csv"
        df.to_csv(output_path, index=False)

        self.logger.info(f"Saved evaluation metrics to {output_path}")

        self._print_summary(df)
        self._plot_results(df)

    def _log_metrics(self, model_name, metrics):
        self.logger.info(f"{model_name} results:")
        for k, v in metrics.items():
            if k != "model" and v is not None:
                self.logger.info(f"  {k}: {v:.4f}")

    def _print_summary(self, df):
        print("\n" + "=" * 80)
        print("MODEL EVALUATION RESULTS")
        print("=" * 80)
        print(df.to_string(index=False))

    def _plot_results(self, df):
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

        plt.figure(figsize=(12, 7))
        x = np.arange(len(df["model"]))
        width = 0.15

        for i, metric in enumerate(metrics):
            if metric in df.columns:
                plt.bar(x + i * width, df[metric], width, label=metric.upper())

        plt.xlabel("Models")
        plt.ylabel("Score")
        plt.title("Model Performance Comparison")
        plt.xticks(x + width * 2, df["model"], rotation=30)
        plt.legend()
        plt.tight_layout()

        plot_path = PLOTS_DIR / "model_comparison.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        self.logger.info(f"Saved performance plot to {plot_path}")
