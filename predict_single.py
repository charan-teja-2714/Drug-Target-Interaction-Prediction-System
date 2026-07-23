import joblib
import pandas as pd
from src.feature_extraction import FeatureExtractor

# Load trained model
model = joblib.load("models/saved_models/random_forest.joblib")

# New drug–target pair
data = {
    "smiles": ["CC(C)NC1=NC=NC2=C1N=CN2"],
    "protein_sequence": ["MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRK"]
}

df = pd.DataFrame(data)

# Feature extraction (NO labels)
extractor = FeatureExtractor()
X, _ = extractor.extract_features(df, training=False)

# Prediction
pred = model.predict(X)[0]
prob = model.predict_proba(X)[0][1]

print("Prediction:", "Interaction" if pred == 1 else "No Interaction")
print("Confidence:", round(prob, 4))
