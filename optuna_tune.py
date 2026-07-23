#!/usr/bin/env python3
"""
optuna_tune.py

Bayesian hyperparameter optimisation for the best-performing model.

Runs after main.py — uses the same dataset and features, then searches
for optimal LightGBM hyperparameters using Optuna (TPE sampler).

Usage:
    python optuna_tune.py

Output:
    - Best hyperparameters printed to console
    - Best model saved to models/saved_models/lightgbm_optuna.joblib
    - Results saved to results/optuna_results.csv
"""

import logging
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold

import optuna
from optuna.samplers import TPESampler
from lightgbm import LGBMClassifier

from src.data_loader import DataLoader
from src.feature_extraction import FeatureExtractor
from src.config import RANDOM_STATE, TEST_SIZE, SAVED_MODELS_DIR, RESULTS_DIR
from src.utils import setup_logging

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_features():
    """Load dataset and extract features (reuses existing pipeline)."""
    logging.info("Loading dataset...")
    data_loader = DataLoader()
    df = data_loader.load_data()

    logging.info("Extracting features...")
    extractor = FeatureExtractor()
    X, y = extractor.extract_features(df, training=True)

    return X, y


def objective(trial, X_train, y_train):
    """Optuna objective function — maximise 5-fold CV AUC."""

    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 300, 1500),
        "max_depth":         trial.suggest_int("max_depth", 4, 12),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.4, 0.9),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 0.9),
        "bagging_freq":      trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        "n_jobs":            -1,
        "verbose":           -1,
        "random_state":      RANDOM_STATE,
    }

    model  = LGBMClassifier(**params)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Load features
    # ------------------------------------------------------------------
    X, y = load_features()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Remove zero-variance features (same step as main pipeline)
    selector = VarianceThreshold(threshold=0.0)
    X_train  = selector.fit_transform(X_train)
    X_test   = selector.transform(X_test)
    logger.info(f"Feature dimensions after selection: {X_train.shape[1]}")

    # ------------------------------------------------------------------
    # Run Optuna optimisation
    # ------------------------------------------------------------------
    N_TRIALS = 80   # increase to 150+ for better results (takes longer)

    logger.info(f"Starting Optuna optimisation ({N_TRIALS} trials)...")
    print(f"\n{'='*60}")
    print(f"  Optuna Bayesian Optimisation — LightGBM")
    print(f"  {N_TRIALS} trials × 5-fold CV (objective: ROC-AUC)")
    print(f"{'='*60}\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
        study_name="lgbm_dti_optimisation",
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_auc    = study.best_value
    logger.info(f"Best CV ROC-AUC: {best_auc:.4f}")
    logger.info(f"Best params: {best_params}")

    # ------------------------------------------------------------------
    # Retrain on full training set with best params
    # ------------------------------------------------------------------
    print(f"\nRetraining with best hyperparameters...")
    best_params.update({"n_jobs": -1, "verbose": -1, "random_state": RANDOM_STATE})
    best_model = LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train)

    # Evaluate on held-out test set
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    y_pred  = best_model.predict(X_test)
    y_prob  = best_model.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"  OPTUNA-TUNED LIGHTGBM — TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Save model and results
    # ------------------------------------------------------------------
    model_path = SAVED_MODELS_DIR / "lightgbm_optuna.joblib"
    joblib.dump(best_model, model_path)
    logger.info(f"Saved optimised model to {model_path}")

    results_df = pd.DataFrame([{
        "model":    "lightgbm_optuna",
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc":  auc,
        **best_params
    }])
    results_path = RESULTS_DIR / "optuna_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved Optuna results to {results_path}")

    print(f"Best hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
