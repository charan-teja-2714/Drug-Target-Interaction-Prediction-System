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
from sklearn.model_selection import train_test_split

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

        models = get_all_models()
        trained_models = {}

        for model_name, model in models.items():
            self.logger.info(f"Training model: {model_name}")

            try:
                model.fit(X_train, y_train)

                # Save model
                model_path = SAVED_MODELS_DIR / f"{model_name}_tuned.joblib"
                joblib.dump(model, model_path)

                self.logger.info(f"Saved {model_name} to {model_path}")

                trained_models[model_name] = {
                    "model": model,
                    "X_test": X_test,
                    "y_test": y_test
                }

            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {str(e)}")

        self.logger.info(f"Training completed for {len(trained_models)} models")
        return trained_models
