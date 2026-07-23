# """
# Configuration settings for DTI Prediction Application
# """

# import os
# from pathlib import Path

# class Config:
#     """Central configuration class"""
    
#     def __init__(self):
#         # Project paths
#         self.PROJECT_ROOT = Path(__file__).parent.parent
#         self.DATA_DIR = self.PROJECT_ROOT / "data"
#         self.RAW_DATA_DIR = self.DATA_DIR / "raw"
#         self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
#         self.MODELS_DIR = self.PROJECT_ROOT / "models" / "saved_models"
#         self.RESULTS_DIR = self.PROJECT_ROOT / "results"
#         self.PLOTS_DIR = self.RESULTS_DIR / "plots"
        
#         # Dataset configuration
#         self.INPUT_FILE = self.RAW_DATA_DIR / "bindingdb_filtered.csv"
#         self.REQUIRED_COLUMNS = ['smiles', 'protein_sequence', 'interaction']
        
#         # Feature extraction parameters
#         self.MORGAN_RADIUS = 2
#         self.MORGAN_NBITS = 1024
        
#         # Model parameters
#         self.RANDOM_STATE = 42
#         self.TEST_SIZE = 0.2
        
#         # Model hyperparameters
#         self.MODEL_PARAMS = {
#             'logistic_regression': {
#                 'random_state': self.RANDOM_STATE,
#                 'max_iter': 1000
#             },
#             'random_forest': {
#                 'n_estimators': 100,
#                 'random_state': self.RANDOM_STATE,
#                 'n_jobs': -1
#             },
#             'svm': {
#                 'random_state': self.RANDOM_STATE,
#                 'probability': True
#             },
#             'xgboost': {
#                 'random_state': self.RANDOM_STATE,
#                 'eval_metric': 'logloss'
#             },
#             'neural_network': {
#                 'hidden_layer_sizes': (100, 50),
#                 'random_state': self.RANDOM_STATE,
#                 'max_iter': 500
#             }
#         }
        
#         # Output files
#         self.METRICS_FILE = self.RESULTS_DIR / "metrics.csv"

"""
config.py

Central configuration file for the Drug–Target Interaction (DTI) Prediction project.
All paths, constants, and model hyperparameters are defined here to ensure
reproducibility and maintainability.

Compatible with:
- OS: Windows
- Python: 3.10
"""

import os
from pathlib import Path

# ------------------------------------------------------------------
# Project Root Directory
# ------------------------------------------------------------------
# Assumes this file is located at: DTI_Prediction/src/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------
# Data Paths
# ------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / "bindingdb_balanced_20000.csv"

# ------------------------------------------------------------------
# Model & Results Paths
# ------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create directories if they do not exist
for path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
]:
    os.makedirs(path, exist_ok=True)

# ------------------------------------------------------------------
# Random Seed (for reproducibility)
# ------------------------------------------------------------------
RANDOM_STATE = 42

# ------------------------------------------------------------------
# Dataset Columns (Expected)
# ------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "smiles",
    "protein_sequence",
    "interaction",
]

# ------------------------------------------------------------------
# Feature Extraction Parameters
# ------------------------------------------------------------------
# Drug (Morgan Fingerprints)
MORGAN_RADIUS = 2
MORGAN_NBITS = 1024

# Protein Features
AMINO_ACIDS = [
    "A", "C", "D", "E", "F",
    "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R",
    "S", "T", "V", "W", "Y"
]

# ------------------------------------------------------------------
# Train-Test Split
# ------------------------------------------------------------------
TEST_SIZE = 0.2

# ------------------------------------------------------------------
# Model Hyperparameters
# ------------------------------------------------------------------

# Logistic Regression
LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

# Random Forest
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Support Vector Machine
SVM_PARAMS = {
    "kernel": "rbf",
    "probability": True,
    "random_state": RANDOM_STATE,
}

# XGBoost
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
}

# Neural Network (MLPClassifier)
MLP_PARAMS = {
    "hidden_layer_sizes": (256, 128),
    "activation": "relu",
    "solver": "adam",
    "max_iter": 300,
    "random_state": RANDOM_STATE,
}

# ------------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------------
METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
]
