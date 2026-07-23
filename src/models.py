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



"""
models.py  –  Enhanced v2

Models added / improved:
  - LightGBM (new, typically faster & more accurate than XGBoost)
  - Neural Network upgraded: (512, 256, 128) + early stopping
  - Logistic Regression, SVM, Neural Network wrapped in StandardScaler
    pipelines (critical for the new mixed-scale feature set)
  - Stacking Ensemble: RF + XGBoost + LightGBM → Logistic Regression
    meta-learner (new, highest expected accuracy)
"""

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier, VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    from catboost import CatBoostClassifier
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False


def get_all_models(random_state=42):
    """
    Initialize and return all ML models.

    Tree-based models (RF, XGBoost, LightGBM) work directly on raw features.
    Linear and distance-based models (LR, SVM, MLP) are wrapped in a
    StandardScaler Pipeline to handle the mixed-scale feature set.

    Returns
    -------
    dict  –  model_name -> model_instance
    """

    # ------------------------------------------------------------------
    # Tuned hyperparameters for the ~2158-feature set
    # (interaction features added, variance-filtered)
    # ------------------------------------------------------------------

    # RF: more trees + lower max_depth prevents overfitting on larger feature set
    _rf_params = dict(
        n_estimators=500,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",      # sqrt(2158) ≈ 46 — standard for classification
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )

    # XGBoost: lower colsample + reg helps with high-dimensional sparse features
    _xgb_params = dict(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,     # sample fewer features per tree
        colsample_bylevel=0.8,
        reg_alpha=0.1,            # L1 regularisation
        reg_lambda=1.0,           # L2 regularisation
        eval_metric="logloss",
        n_jobs=-1,
        random_state=random_state,
    )

    # LightGBM: deeper tree search with stronger regularisation
    _lgbm_params = dict(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.03,       # slower learning → better generalisation
        num_leaves=127,           # 127 vs 63: captures finer patterns
        feature_fraction=0.6,
        bagging_fraction=0.8,
        bagging_freq=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=20,
        n_jobs=-1,
        verbose=-1,
        random_state=random_state,
    )

    # MLP params reused inside stacking base estimator
    _mlp_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
        )),
    ])

    # ---------------------------------------------------------------
    # Fixed Stacking Ensemble
    # Base: RF (trees) + XGBoost (boosting) + LightGBM (boosting)
    #       + MLP (neural net) → diverse bias/variance profiles
    # Meta-learner: LightGBM (non-linear, better than LR)
    # passthrough=True: meta-learner also sees original features
    # ---------------------------------------------------------------
    stacking = StackingClassifier(
        estimators=[
            ("rf",   RandomForestClassifier(**_rf_params)),
            ("xgb",  XGBClassifier(**_xgb_params)),
            ("lgbm", LGBMClassifier(**_lgbm_params)),
            ("mlp",  _mlp_pipe),
        ],
        final_estimator=LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            verbose=-1,
            random_state=random_state,
        ),
        cv=3,
        n_jobs=-1,
        passthrough=True,         # feed original features to meta-learner too
    )

    # ---------------------------------------------------------------
    # Voting Ensemble: soft probability averaging of top 3 models
    # Simple but robust — no meta-learning, just averaged confidence
    # ---------------------------------------------------------------
    voting = VotingClassifier(
        estimators=[
            ("rf",   RandomForestClassifier(**_rf_params)),
            ("xgb",  XGBClassifier(**_xgb_params)),
            ("lgbm", LGBMClassifier(**_lgbm_params)),
        ],
        voting="soft",
        n_jobs=-1,
    )

    return {
        # Linear model — needs scaling
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(
                C=0.5, max_iter=2000, random_state=random_state
            )),
        ]),

        # Tree ensemble — no scaling needed
        "random_forest": RandomForestClassifier(**_rf_params),

        # SVM — needs scaling; NOTE: slowest model, may take 20-40 min
        # "svm": Pipeline([
        #     ("scaler", StandardScaler()),
        #     ("model",  SVC(kernel="rbf", C=1.0, probability=True,
        #                    random_state=random_state)),
        # ]),

        # Gradient boosting — no scaling needed
        "xgboost": XGBClassifier(**_xgb_params),

        # LightGBM — faster & often more accurate than XGBoost
        "lightgbm": LGBMClassifier(**_lgbm_params),

        # Deep MLP — needs scaling; (512,256,128) with early stopping
        "neural_network": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MLPClassifier(
                hidden_layer_sizes=(512, 256, 128),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=random_state,
            )),
        ]),

        # ExtraTrees: more randomised than RF — fast, good diversity for ensembles
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),

        # HistGradientBoosting: sklearn's native fast boosting (no extra install)
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=8,
            learning_rate=0.05,
            min_samples_leaf=20,
            l2_regularization=0.1,
            max_leaf_nodes=63,
            random_state=random_state,
        ),

        # Voting ensemble — soft average of RF + XGBoost + LightGBM
        "voting": voting,

        # Stacking ensemble — highest expected accuracy
        "stacking": stacking,

        # CatBoost — included if installed (pip install catboost)
        **({
            "catboost": CatBoostClassifier(
                iterations=700,
                depth=8,
                learning_rate=0.05,
                l2_leaf_reg=3,
                random_seed=random_state,
                verbose=0,
            )
        } if _CATBOOST_AVAILABLE else {}),
    }
