# """
# Model training module
# """

# import joblib
# import logging
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# from src.models import ModelFactory
# from src.preprocessing import DataPreprocessor

# class ModelTrainer:
#     """Handles model training and saving"""
    
#     def __init__(self, config):
#         self.config = config
#         self.logger = logging.getLogger(__name__)
#         self.model_factory = ModelFactory(config)
#         self.preprocessor = DataPreprocessor(config)
    
#     def train_all_models(self, X, y):
#         """
#         Train all models
        
#         Args:
#             X: Feature matrix
#             y: Target vector
            
#         Returns:
#             dict: Dictionary of trained models with train/test splits
#         """
#         self.logger.info("Starting model training...")
        
#         # Split data
#         X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        
#         # Create models
#         models = self.model_factory.create_all_models()
        
#         # Train each model
#         trained_models = {}
        
#         for model_name, model in models.items():
#             self.logger.info(f"Training {model_name}...")
            
#             try:
#                 # Train model
#                 model.fit(X_train, y_train)
                
#                 # Save trained model
#                 self.save_model(model, model_name)
                
#                 # Store model with data splits for evaluation
#                 trained_models[model_name] = {
#                     'model': model,
#                     'X_train': X_train,
#                     'X_test': X_test,
#                     'y_train': y_train,
#                     'y_test': y_test
#                 }
                
#                 self.logger.info(f"Successfully trained {model_name}")
                
#             except Exception as e:
#                 self.logger.error(f"Failed to train {model_name}: {str(e)}")
#                 continue
        
#         self.logger.info(f"Training completed for {len(trained_models)} models")
#         return trained_models
    
#     def save_model(self, model, model_name):
#         """
#         Save trained model to disk
        
#         Args:
#             model: Trained model
#             model_name: Name of the model
#         """
#         # Create filename
#         filename = f"{model_name.lower().replace(' ', '_')}.joblib"
#         filepath = self.config.MODELS_DIR / filename
        
#         # Ensure directory exists
#         filepath.parent.mkdir(parents=True, exist_ok=True)
        
#         # Save model
#         joblib.dump(model, filepath)
#         self.logger.info(f"Saved {model_name} to {filepath}")
    
#     def load_model(self, model_name):
#         """
#         Load saved model from disk
        
#         Args:
#             model_name: Name of the model
            
#         Returns:
#             Loaded model
#         """
#         filename = f"{model_name.lower().replace(' ', '_')}.joblib"
#         filepath = self.config.MODELS_DIR / filename
        
#         if not filepath.exists():
#             raise FileNotFoundError(f"Model file not found: {filepath}")
        
#         model = joblib.load(filepath)
#         self.logger.info(f"Loaded {model_name} from {filepath}")
#         return model


"""
train.py

Model training module for DTI Prediction.
Trains all machine learning models and saves them to disk.
"""

import logging
import joblib
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    SAVED_MODELS_DIR
)
from src.models import get_all_models


class ModelTrainer:
    """Handles model training and saving"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def train_all_models(self, X, y):
        """
        Train all models and save them.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels

        Returns
        -------
        dict
            Dictionary of trained models
        """

        self.logger.info("Starting model training...")

        # --------------------------------------------------
        # Train-test split
        # --------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        # --------------------------------------------------
        # Feature selection: remove zero-variance features
        # (Morgan/MACCS bits that are constant across the dataset
        #  carry no information and add noise for SVM/MLP)
        # --------------------------------------------------
        selector = VarianceThreshold(threshold=0.0)
        X_train = selector.fit_transform(X_train)
        X_test  = selector.transform(X_test)
        joblib.dump(selector, SAVED_MODELS_DIR / "variance_selector.joblib")
        self.logger.info(
            f"Feature selection: kept {X_train.shape[1]} / "
            f"{selector.n_features_in_} features "
            f"(removed {selector.n_features_in_ - X_train.shape[1]} zero-variance)"
        )

        models = get_all_models()
        trained_models = {}
        model_names = list(models.keys())

        print(f"\n  Training {len(model_names)} models on "
              f"{X_train.shape[0]:,} samples × {X_train.shape[1]:,} features\n")

        pbar = tqdm(model_names, desc="Models", unit="model",
                    ncols=80, leave=True)

        for model_name in pbar:
            model = models[model_name]
            pbar.set_description(f"Training  {model_name:<28}")

            try:
                t0 = time.time()
                model.fit(X_train, y_train)
                elapsed = time.time() - t0

                # Save model
                model_path = SAVED_MODELS_DIR / f"{model_name}_tuned.joblib"
                joblib.dump(model, model_path)

                pbar.write(f"  ✔  {model_name:<28}  done in {elapsed:6.1f}s")
                self.logger.info(f"Saved {model_name} to {model_path}")

                trained_models[model_name] = {
                    "model": model,
                    "X_test": X_test,
                    "y_test": y_test
                }

            except Exception as e:
                pbar.write(f"  ✘  {model_name:<28}  FAILED: {e}")
                self.logger.error(f"Failed to train {model_name}: {str(e)}")

        print()
        self.logger.info(f"Training completed for {len(trained_models)} models")
        return trained_models
