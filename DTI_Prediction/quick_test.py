import joblib
import pandas as pd
from src.feature_extraction import FeatureExtractor

# Load trained model
model = joblib.load("models/saved_models/logistic_regression.joblib")

# Load small sample
df = pd.read_csv("data/raw/bindingdb_balanced_20000.csv").head(5)

# Extract features
extractor = FeatureExtractor()
X, y = extractor.extract_features(df)

# Predict
preds = model.predict(X)
probs = model.predict_proba(X)[:, 1]

print("True labels:", y)
print("Predictions:", preds)
print("Probabilities:", probs)
