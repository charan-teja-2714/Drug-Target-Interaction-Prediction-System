# """
# Machine learning models module
# """

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# import xgboost as xgb
# import logging

# class ModelFactory:
#     """Factory class for creating ML models"""
    
#     def __init__(self, config):
#         self.config = config
#         self.logger = logging.getLogger(__name__)
    
#     def create_logistic_regression(self):
#         """
#         Create Logistic Regression model
        
#         Returns:
#             LogisticRegression: Configured model
#         """
#         params = self.config.MODEL_PARAMS['logistic_regression']
#         return LogisticRegression(**params)
    
#     def create_random_forest(self):
#         """
#         Create Random Forest model
        
#         Returns:
#             RandomForestClassifier: Configured model
#         """
#         params = self.config.MODEL_PARAMS['random_forest']
#         return RandomForestClassifier(**params)
    
#     def create_svm(self):
#         """
#         Create Support Vector Machine model
        
#         Returns:
#             SVC: Configured model
#         """
#         params = self.config.MODEL_PARAMS['svm']
#         return SVC(**params)
    
#     def create_xgboost(self):
#         """
#         Create XGBoost model
        
#         Returns:
#             XGBClassifier: Configured model
#         """
#         params = self.config.MODEL_PARAMS['xgboost']
#         return xgb.XGBClassifier(**params)
    
#     def create_neural_network(self):
#         """
#         Create Neural Network (MLP) model
        
#         Returns:
#             MLPClassifier: Configured model
#         """
#         params = self.config.MODEL_PARAMS['neural_network']
#         return MLPClassifier(**params)
    
#     def create_all_models(self):
#         """
#         Create all models
        
#         Returns:
#             dict: Dictionary of model name -> model instance
#         """
#         models = {
#             'Logistic Regression': self.create_logistic_regression(),
#             'Random Forest': self.create_random_forest(),
#             'SVM': self.create_svm(),
#             'XGBoost': self.create_xgboost(),
#             'Neural Network': self.create_neural_network()
#         }
        
#         self.logger.info(f"Created {len(models)} models: {list(models.keys())}")
#         return models


"""
models.py

Defines all machine learning models used for
Drug–Target Interaction (DTI) prediction.
"""

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier

# from src.config import (
#     RANDOM_STATE,
#     LOGISTIC_REGRESSION_PARAMS,
#     RANDOM_FOREST_PARAMS,
#     SVM_PARAMS,
#     XGBOOST_PARAMS,
#     MLP_PARAMS
# )


# def get_all_models():
#     """
#     Initialize and return all ML models.

#     Returns
#     -------
#     dict
#         Dictionary of model_name -> model_instance
#     """

#     models = {
#         "logistic_regression": LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),

#         "random_forest": RandomForestClassifier(**RANDOM_FOREST_PARAMS),

#         "svm": SVC(**SVM_PARAMS),

#         "xgboost": XGBClassifier(
#             tree_method="gpu_hist",
#             predictor="gpu_predictor",
#             eval_metric="logloss",
#             random_state=42
#         ),

#         "neural_network": MLPClassifier(**MLP_PARAMS),
#     }

#     return models



from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def get_all_models(random_state=42):
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        ),

        "svm": SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state
        ),

        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        ),

        "neural_network": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=random_state
        )
    }
