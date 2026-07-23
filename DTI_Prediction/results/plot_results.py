import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.feature_extraction import FeatureExtractor

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_FILE = "data/raw/bindingdb_balanced_20000.csv"
MODELS_DIR = "models/saved_models"
PLOTS_DIR = "results/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

extractor = FeatureExtractor()
X, y = extractor.extract_features(df)

# --------------------------------------------------
# Loop through models
# --------------------------------------------------
for model_file in os.listdir(MODELS_DIR):
    if not model_file.endswith(".joblib"):
        continue

    model_name = model_file.replace(".joblib", "")
    print(f"Processing model: {model_name}")

    model = joblib.load(os.path.join(MODELS_DIR, model_file))

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        # For models without probability (rare)
        y_prob = y_pred

    # --------------------------------------------------
    # ROC Curve
    # --------------------------------------------------
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {model_name.replace('_', ' ').title()}")
    plt.legend(loc="lower right")
    plt.grid(True)

    roc_path = os.path.join(PLOTS_DIR, f"roc_{model_name}.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    cm = confusion_matrix(y, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Interaction", "Interaction"]
    )
    disp.plot(cmap="Blues", values_format="d")

    plt.title(f"Confusion Matrix – {model_name.replace('_', ' ').title()}")

    cm_path = os.path.join(PLOTS_DIR, f"cm_{model_name}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

print("✅ ROC curves and confusion matrices generated successfully")
