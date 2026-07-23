# Technical Documentation: Drug-Target Interaction Prediction System

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts Explained](#core-concepts-explained)
3. [Dataset Information](#dataset-information)
4. [System Architecture](#system-architecture)
5. [Feature Engineering](#feature-engineering)
6. [Machine Learning Models](#machine-learning-models)
7. [Implementation Details](#implementation-details)
8. [Performance Results](#performance-results)
9. [Usage Guide](#usage-guide)

---

## 1. Introduction

### What is Drug-Target Interaction (DTI) Prediction?

**Drug-Target Interaction** refers to the binding relationship between a drug molecule (small chemical compound) and a biological target (usually a protein). Understanding these interactions is crucial for:

- **Drug Discovery**: Identifying which drugs might work on specific disease targets
- **Drug Repurposing**: Finding new uses for existing drugs
- **Side Effect Prediction**: Understanding unintended drug-protein interactions
- **Cost Reduction**: Computational screening is cheaper than laboratory experiments

### The Problem

Traditional experimental methods to test drug-target interactions are:
- **Expensive**: Each experiment costs thousands of dollars
- **Time-consuming**: Can take weeks or months
- **Limited scale**: Cannot test millions of combinations

### Our Solution

A machine learning system that predicts whether a drug will interact with a protein target based on their molecular structures, providing:
- **Fast predictions**: Results in milliseconds
- **Scalable**: Can screen thousands of combinations
- **Probabilistic**: Provides confidence scores

---

## 2. Core Concepts Explained

### 2.1 SMILES Notation

**SMILES** (Simplified Molecular Input Line Entry System) is a text-based representation of chemical structures.

**Example**:
```
Aspirin: CC(=O)Oc1ccccc1C(=O)O
Caffeine: CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

**How it works**:
- Letters represent atoms (C=Carbon, N=Nitrogen, O=Oxygen)
- Numbers represent ring structures
- Parentheses show branches
- `=` represents double bonds

### 2.2 Protein Sequences

Proteins are chains of amino acids represented by single letters.

**Example**:
```
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRK
```

**20 Standard Amino Acids**:
- A (Alanine), C (Cysteine), D (Aspartic acid), E (Glutamic acid)
- F (Phenylalanine), G (Glycine), H (Histidine), I (Isoleucine)
- K (Lysine), L (Leucine), M (Methionine), N (Asparagine)
- P (Proline), Q (Glutamine), R (Arginine), S (Serine)
- T (Threonine), V (Valine), W (Tryptophan), Y (Tyrosine)

### 2.3 Binary Classification

Our system performs **binary classification**:
- **Class 0**: No interaction between drug and protein
- **Class 1**: Interaction exists between drug and protein

### 2.4 Morgan Fingerprints

**Morgan Fingerprints** (also called Circular Fingerprints) convert molecular structures into fixed-length binary vectors.

**How it works**:
1. Start from each atom in the molecule
2. Look at neighboring atoms within a certain radius
3. Generate a unique identifier for each local structure
4. Hash these identifiers into a fixed-length bit vector

**Parameters**:
- **Radius**: How many bonds away to look (we use radius=2)
- **nBits**: Length of the fingerprint vector (we use 1024 bits)

**Example**:
```
Molecule → [0,1,0,0,1,1,0,1,0,0,...] (1024 bits)
```

### 2.5 Amino Acid Composition (AAC)

**AAC** calculates the frequency of each amino acid in a protein sequence.

**Example**:
```
Sequence: MTEYKLVVVGAGGVGK (16 amino acids)
- M appears 1 time → 1/16 = 0.0625
- T appears 1 time → 1/16 = 0.0625
- E appears 1 time → 1/16 = 0.0625
- Y appears 1 time → 1/16 = 0.0625
- K appears 2 times → 2/16 = 0.1250
- L appears 1 time → 1/16 = 0.0625
- V appears 4 times → 4/16 = 0.2500
- G appears 4 times → 4/16 = 0.2500
- A appears 1 time → 1/16 = 0.0625
```

Result: 20-dimensional vector (one for each amino acid)

### 2.6 CTD Descriptors

**CTD** (Composition, Transition, Distribution) groups amino acids by physicochemical properties.

**7 Property Groups**:
1. **Hydrophobicity**: How water-repelling the amino acid is
2. **Van der Waals Volume**: Physical size of the amino acid
3. **Polarity**: Electric charge distribution
4. **Polarizability**: How easily electrons can be distorted
5. **Charge**: Positive, negative, or neutral
6. **Secondary Structure**: Tendency to form helices, strands, or coils
7. **Solvent Accessibility**: How exposed to water the amino acid is

Each property has 3 subgroups, giving 21 composition features (7 × 3).

---

## 3. Dataset Information

### 3.1 BindingDB

**BindingDB** is a public database containing measured binding affinities between drugs and proteins.

**Source**: https://www.bindingdb.org/

**Contains**:
- Drug molecules (as SMILES)
- Protein sequences
- Binding affinity measurements (Ki, Kd, IC50, EC50)

### 3.2 Our Dataset

**File**: `data/raw/bindingdb_balanced_20000.csv`

**Size**: 20,000 drug-protein pairs
- 10,000 positive interactions (interaction = 1)
- 10,000 negative interactions (interaction = 0)

**Columns**:
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `smiles` | String | Drug structure in SMILES format | `CC(C)NC1=NC=NC2=C1N=CN2` |
| `protein_sequence` | String | Protein amino acid sequence | `MTEYKLVVVGAGGVGK...` |
| `interaction` | Integer | Binary label (0 or 1) | `1` |

### 3.3 Data Balancing

**Why balance?**
- Original BindingDB has more negative samples than positive
- Imbalanced data causes models to be biased toward the majority class
- Balanced data ensures fair learning

**How we balanced**:
```python
# balance_dataset.py
- Sample 10,000 positive interactions
- Sample 10,000 negative interactions
- Shuffle and combine
```

### 3.4 Interaction Labels

**How labels were created**:
- Binding affinity ≤ 1000 nM → **Interaction (1)**
- Binding affinity > 1000 nM → **No Interaction (0)**

**Why 1000 nM threshold?**
- Standard cutoff in pharmaceutical research
- Represents "strong enough" binding for drug effect

---

## 4. System Architecture

### 4.1 Pipeline Overview

```
┌─────────────────┐
│  Raw Dataset    │
│  (CSV file)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Loader    │ ← Validation, cleaning, deduplication
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature         │ ← Drug: Morgan fingerprints (1024-dim)
│ Extraction      │ ← Protein: AAC + CTD (41-dim)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Train/Test      │ ← 80/20 stratified split
│ Split           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │ ← 5 ML models trained in parallel
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model           │ ← Metrics calculation
│ Evaluation      │ ← Visualization generation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Results Export  │ ← CSV metrics, plots, saved models
└─────────────────┘
```

### 4.2 Module Breakdown

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `config.py` | Central configuration | Paths, hyperparameters, constants |
| `data_loader.py` | Data loading & validation | Load CSV, validate schema, clean data |
| `feature_extraction.py` | Feature engineering | Morgan fingerprints, AAC, CTD |
| `models.py` | Model definitions | Initialize 5 ML models |
| `train.py` | Model training | Train/test split, fit models, save |
| `evaluate.py` | Performance evaluation | Calculate metrics, generate plots |
| `utils.py` | Utility functions | Logging setup |
| `main.py` | Pipeline orchestration | Execute complete workflow |

### 4.3 Data Flow

**Input**: CSV with 3 columns (smiles, protein_sequence, interaction)

**Processing**:
1. Load 20,000 rows
2. Extract 1024 drug features + 41 protein features = **1065 total features**
3. Split: 16,000 training samples + 4,000 test samples
4. Train 5 models on training data
5. Evaluate on test data

**Output**:
- 5 trained model files (.joblib)
- Performance metrics (CSV)
- Comparison plots (PNG)

---

## 5. Feature Engineering

### 5.1 Drug Features (1024 dimensions)

**Method**: Morgan Fingerprints using RDKit

**Code**:
```python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

mol = Chem.MolFromSmiles(smiles)
fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
    mol, 
    radius=2,      # Look 2 bonds away
    nBits=1024     # 1024-bit vector
)
```

**Output**: Binary vector [0,1,0,1,1,0,...] of length 1024

**Invalid SMILES**: Replaced with zero vector [0,0,0,...,0]

### 5.2 Protein Features (41 dimensions)

#### Part 1: Amino Acid Composition (20 dimensions)

**Code**:
```python
from collections import Counter

def amino_acid_composition(sequence):
    counts = Counter(sequence)
    total = len(sequence)
    return [counts.get(aa, 0) / total for aa in AMINO_ACIDS]
```

**Output**: [freq_A, freq_C, freq_D, ..., freq_Y]

#### Part 2: CTD Composition (21 dimensions)

**Code**:
```python
groups = {
    "hydrophobicity": ["RKEDQN", "GASTPHY", "CLVIMFW"],
    "vdw_volume": ["GASTPDC", "NVEQIL", "MHKFRYW"],
    # ... 7 properties × 3 subgroups = 21 features
}

for group in groups.values():
    for subset in group:
        count = sum(sequence.count(aa) for aa in subset)
        features.append(count / len(sequence))
```

**Output**: 21 composition ratios

### 5.3 Combined Feature Vector

**Final representation**:
```
Drug-Protein Pair → [1024 drug features | 41 protein features]
                     = 1065-dimensional vector
```

---

## 6. Machine Learning Models

### 6.1 Model Selection Rationale

We use 5 different models to compare performance across different learning paradigms:

| Model | Type | Strengths | Use Case |
|-------|------|-----------|----------|
| Logistic Regression | Linear | Fast, interpretable | Baseline comparison |
| Random Forest | Ensemble | Handles non-linearity, robust | Best overall performer |
| SVM | Kernel-based | Good with high dimensions | Complex boundaries |
| XGBoost | Gradient boosting | High accuracy, fast | Production deployment |
| Neural Network | Deep learning | Learns complex patterns | Non-linear relationships |

### 6.2 Hyperparameters (Tuned)

#### Logistic Regression
```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

#### Random Forest
```python
RandomForestClassifier(
    n_estimators=300,      # 300 decision trees
    max_depth=25,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split node
    min_samples_leaf=2,    # Minimum samples in leaf
    class_weight="balanced",  # Handle class imbalance
    n_jobs=-1,             # Use all CPU cores
    random_state=42
)
```

#### Support Vector Machine
```python
SVC(
    kernel="rbf",          # Radial basis function kernel
    probability=True,      # Enable probability estimates
    random_state=42
)
```

#### XGBoost
```python
XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,         # Use 80% of data per tree
    colsample_bytree=0.8,  # Use 80% of features per tree
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
```

#### Neural Network
```python
MLPClassifier(
    hidden_layer_sizes=(256, 128),  # Two hidden layers
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42
)
```

### 6.3 Training Process

**Train/Test Split**:
- 80% training (16,000 samples)
- 20% testing (4,000 samples)
- Stratified split (maintains class balance)

**Training**:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

model.fit(X_train, y_train)
```

**No cross-validation**: Single split for simplicity (academic project)

---

## 7. Implementation Details

### 7.1 File Structure

```
DTI_Prediction/
├── data/
│   ├── raw/
│   │   ├── BindingDB_All.tsv              # Original BindingDB
│   │   ├── bindingdb_filtered.csv         # Filtered version
│   │   └── bindingdb_balanced_20000.csv   # Balanced dataset (USED)
│   └── processed/                         # (Empty - for future use)
│
├── src/
│   ├── config.py                          # Configuration
│   ├── data_loader.py                     # Data loading
│   ├── feature_extraction.py              # Feature engineering
│   ├── models.py                          # Model definitions
│   ├── train.py                           # Training logic
│   ├── evaluate.py                        # Evaluation logic
│   └── utils.py                           # Utilities
│
├── models/
│   └── saved_models/
│       ├── logistic_regression_tuned.joblib
│       ├── random_forest_tuned.joblib
│       ├── svm_tuned.joblib
│       ├── xgboost_tuned.joblib
│       └── neural_network_tuned.joblib
│
├── results/
│   ├── metrics.csv                        # Performance metrics
│   └── plots/
│       ├── model_comparison.png           # Bar chart comparison
│       ├── cm_*.png                       # Confusion matrices
│       └── roc_*.png                      # ROC curves
│
├── main.py                                # Main pipeline
├── predict_single.py                      # Single prediction script
├── balance_dataset.py                     # Dataset balancing script
├── requirements.txt                       # Dependencies
└── README.md                              # User guide
```

### 7.2 Key Implementation Choices

**CPU-Only Processing**:
- All models run on CPU
- No GPU acceleration (not needed for 20k samples)
- Scikit-learn and XGBoost CPU implementations

**Error Handling**:
- Invalid SMILES → Zero vector (no crash)
- Missing values → Dropped during loading
- Failed model training → Logged and skipped

**Reproducibility**:
- Random seed = 42 everywhere
- Deterministic train/test split
- Fixed hyperparameters

**Model Persistence**:
- Models saved as `.joblib` files
- Can be loaded for inference without retraining

---

## 8. Performance Results

### 8.1 Evaluation Metrics Explained

**Accuracy**: Percentage of correct predictions
```
Accuracy = (True Positives + True Negatives) / Total Samples
```

**Precision**: Of all predicted interactions, how many were correct?
```
Precision = True Positives / (True Positives + False Positives)
```

**Recall**: Of all actual interactions, how many did we find?
```
Recall = True Positives / (True Positives + False Negatives)
```

**F1-Score**: Harmonic mean of precision and recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**ROC-AUC**: Area under the ROC curve (0.5 = random, 1.0 = perfect)

### 8.2 Actual Results

**From `results/metrics.csv`**:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **79.8%** | **79.9%** | **79.8%** | **79.8%** | **88.3%** |
| XGBoost | 78.3% | 78.2% | 78.6% | 78.4% | 85.9% |
| SVM | 74.0% | 73.7% | 74.5% | 74.1% | 81.1% |
| Neural Network | 71.9% | 71.8% | 72.0% | 71.9% | 79.5% |
| Logistic Regression | 66.4% | 66.2% | 67.2% | 66.7% | 72.3% |

**Best Model**: Random Forest with 79.8% accuracy

### 8.3 Interpretation

**Random Forest wins because**:
- Handles high-dimensional features well (1065 features)
- Robust to noisy data
- Captures non-linear relationships
- Ensemble method reduces overfitting

**Logistic Regression performs worst because**:
- Linear model cannot capture complex drug-protein interactions
- Serves as baseline for comparison

**Practical meaning**:
- ~80% accuracy means 4 out of 5 predictions are correct
- ROC-AUC of 88.3% indicates strong discriminative ability
- Suitable for initial screening, but experimental validation still needed

---

## 9. Usage Guide

### 9.1 Training Pipeline

**Run complete pipeline**:
```cmd
python main.py
```

**What happens**:
1. Loads `bindingdb_balanced_20000.csv`
2. Extracts features (takes ~2-3 minutes)
3. Trains 5 models (takes ~5-10 minutes)
4. Evaluates and saves results

**Output**:
- Console: Progress logs and final metrics
- `results/metrics.csv`: Performance table
- `results/plots/`: Visualization charts
- `models/saved_models/`: Trained models

### 9.2 Single Prediction

**Predict interaction for a new drug-protein pair**:

```python
# predict_single.py
import joblib
import pandas as pd
from src.feature_extraction import FeatureExtractor

# Load trained model
model = joblib.load("models/saved_models/random_forest_tuned.joblib")

# New drug-target pair
data = {
    "smiles": ["CC(C)NC1=NC=NC2=C1N=CN2"],  # Example drug
    "protein_sequence": ["MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRK"]  # Example protein
}

df = pd.DataFrame(data)

# Extract features
extractor = FeatureExtractor()
X, _ = extractor.extract_features(df, training=False)

# Predict
prediction = model.predict(X)[0]
probability = model.predict_proba(X)[0][1]

print("Prediction:", "Interaction" if prediction == 1 else "No Interaction")
print("Confidence:", round(probability, 4))
```

**Run**:
```cmd
python predict_single.py
```

**Output**:
```
Prediction: Interaction
Confidence: 0.8234
```

### 9.3 Dataset Balancing

**If you have a new unbalanced dataset**:

```python
# balance_dataset.py
python balance_dataset.py
```

This creates `bindingdb_balanced_20000.csv` from `bindingdb_filtered.csv`.

---

## 10. Technical Specifications

### 10.1 System Requirements

- **OS**: Windows 10/11
- **Python**: 3.10.x
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for code + models
- **CPU**: Multi-core recommended (for Random Forest parallelization)

### 10.2 Dependencies

```
joblib==1.3.2          # Model serialization
numpy==1.26.4          # Numerical operations
pandas==2.1.4          # Data manipulation
scikit-learn==1.3.2    # ML models
xgboost==2.0.3         # Gradient boosting
matplotlib==3.8.2      # Plotting
seaborn==0.13.0        # Statistical plots
rdkit-pypi==2022.9.5   # Molecular fingerprints
```

### 10.3 Computational Complexity

**Feature Extraction**:
- Drug: O(n × m) where n = samples, m = atoms per molecule
- Protein: O(n × l) where l = sequence length
- Time: ~2-3 minutes for 20,000 samples

**Model Training**:
- Logistic Regression: O(n × d) where d = features (1065)
- Random Forest: O(n × d × log(n) × trees)
- Time: ~5-10 minutes for all 5 models

**Prediction**:
- Single prediction: <100ms
- Batch prediction: ~1 second per 1000 samples

---

## 11. Limitations & Future Work

### 11.1 Current Limitations

1. **Binary Classification Only**: Cannot predict binding strength, only yes/no
2. **No 3D Structure**: Uses sequence only, ignores protein folding
3. **No Drug Metabolism**: Doesn't consider ADME properties
4. **Single Split**: No cross-validation for robust evaluation
5. **CPU-Only**: Could be faster with GPU acceleration
6. **Fixed Features**: Cannot adapt feature extraction

### 11.2 Potential Improvements

- **Multi-class classification**: Predict binding affinity ranges
- **Deep learning**: Use graph neural networks for molecules
- **Ensemble stacking**: Combine predictions from all models
- **Feature selection**: Reduce dimensionality
- **Cross-validation**: 5-fold or 10-fold CV
- **Hyperparameter tuning**: Grid search or Bayesian optimization

---

## 12. Troubleshooting

### Common Issues

**Issue 1**: `FileNotFoundError: bindingdb_balanced_20000.csv`
- **Solution**: Run `balance_dataset.py` first or place dataset in `data/raw/`

**Issue 2**: `RDKit import error`
- **Solution**: `pip install rdkit-pypi`

**Issue 3**: Slow training
- **Solution**: Reduce dataset size or use fewer trees in Random Forest

**Issue 4**: Low accuracy
- **Solution**: Check data quality, balance classes, tune hyperparameters

---

## 13. References

### Databases
- **BindingDB**: https://www.bindingdb.org/
- Public database of drug-target binding affinities

### Libraries
- **RDKit**: https://www.rdkit.org/ (Molecular fingerprints)
- **Scikit-learn**: https://scikit-learn.org/ (ML models)
- **XGBoost**: https://xgboost.readthedocs.io/ (Gradient boosting)

### Concepts
- **Morgan Fingerprints**: Rogers & Hahn (2010), "Extended-Connectivity Fingerprints"
- **CTD Descriptors**: Dubchak et al. (1995), "Prediction of protein folding class"
- **Drug-Target Interaction**: Ezzat et al. (2019), "Drug-Target Interaction Prediction"

---

## 14. Contact & Support

For questions or issues:
1. Check `dti_prediction.log` for detailed error messages
2. Verify Python version: `python --version`
3. Verify dependencies: `pip list`
4. Review this documentation

---

**Last Updated**: 2024
**Version**: 1.0
**Author**: Final Year Project - DTI Prediction System
