# AI-Based Prediction of Drug–Target Interactions Using Machine Learning

A CLI-based Python application for predicting drug-target interactions using molecular fingerprints and protein sequence descriptors, trained on BindingDB data.

## Project Overview

This application implements a complete machine learning pipeline for Drug-Target Interaction (DTI) prediction using:

- **Drug Features**: RDKit Morgan Fingerprints (radius=2, nBits=1024)
- **Protein Features**: Amino Acid Composition + CTD descriptors
- **Models**: Logistic Regression, Random Forest, SVM, XGBoost, Neural Network
- **Dataset**: Filtered BindingDB data with binary interaction labels

## Dataset Information

### BindingDB Dataset
The application expects a pre-filtered CSV file derived from BindingDB with the following structure:

| Column | Description |
|--------|-------------|
| `smiles` | Drug SMILES notation |
| `protein_sequence` | Protein FASTA sequence |
| `interaction` | Binary label (0=no interaction, 1=interaction) |

**Dataset Requirements:**
- Size: ~50,000 rows
- Human proteins only
- Binary labels created using affinity threshold (≤1000 nM → 1)
- File location: `data/raw/bindingdb_filtered.csv`

## Project Structure

```
DTI_Prediction/
│
├── data/
│   ├── raw/
│   │   └── bindingdb_filtered.csv    # Input dataset (user provided)
│   └── processed/                    # Processed data files
│
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── config.py                     # Configuration settings
│   ├── data_loader.py               # Data loading and validation
│   ├── preprocessing.py             # Data preprocessing
│   ├── feature_extraction.py        # Molecular feature extraction
│   ├── models.py                    # ML model definitions
│   ├── train.py                     # Model training
│   ├── evaluate.py                  # Model evaluation
│   └── utils.py                     # Utility functions
│
├── models/
│   └── saved_models/                # Trained model files
│
├── results/
│   ├── metrics.csv                  # Evaluation results
│   └── plots/                       # Result visualizations
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── main.py                         # Main entry point
```

## Setup Instructions (Windows)

### Prerequisites
- Python 3.10 (required)
- Windows operating system

### Installation Steps

1. **Clone or download the project**
   ```cmd
   cd "C:\path\to\your\projects"
   ```

2. **Create virtual environment**
   ```cmd
   python -m venv dti_env
   dti_env\Scripts\activate
   ```

3. **Install dependencies**
   ```cmd
   cd DTI_Prediction
   pip install -r requirements.txt
   ```

4. **Prepare dataset**
   - Place your filtered BindingDB CSV file at: `data/raw/bindingdb_filtered.csv`
   - Ensure the file has columns: `smiles`, `protein_sequence`, `interaction`

## How to Run the Project

### Basic Usage
```cmd
# Activate virtual environment
dti_env\Scripts\activate

# Run the complete pipeline
python main.py
```

### What the Application Does

1. **Data Loading**: Loads and validates the BindingDB dataset
2. **Feature Extraction**: 
   - Extracts Morgan fingerprints from drug SMILES
   - Calculates amino acid composition and CTD descriptors for proteins
3. **Model Training**: Trains all 5 machine learning models
4. **Evaluation**: Computes performance metrics for each model
5. **Results**: Saves metrics and creates visualizations

### Expected Runtime
- Small dataset (~1,000 rows): 1-2 minutes
- Medium dataset (~10,000 rows): 5-10 minutes  
- Large dataset (~50,000 rows): 15-30 minutes

## Expected Outputs

### Console Output
The application will display:
- Data loading statistics
- Feature extraction progress
- Model training progress
- Evaluation results for each model
- Best performing models for each metric

### Generated Files

1. **`results/metrics.csv`**: Performance metrics for all models
   ```csv
   Model,accuracy,precision,recall,f1_score,roc_auc
   Logistic Regression,0.8234,0.7891,0.8456,0.8165,0.8923
   Random Forest,0.8567,0.8234,0.8789,0.8503,0.9156
   ...
   ```

2. **`models/saved_models/`**: Trained model files
   - `logistic_regression.joblib`
   - `random_forest.joblib`
   - `svm.joblib`
   - `xgboost.joblib`
   - `neural_network.joblib`

3. **`results/plots/model_comparison.png`**: Performance comparison chart

4. **`dti_prediction.log`**: Detailed execution log

## Model Information

### Implemented Models
1. **Logistic Regression**: Linear baseline model
2. **Random Forest**: Ensemble method with 100 trees
3. **Support Vector Machine**: RBF kernel with probability estimates
4. **XGBoost**: Gradient boosting classifier
5. **Neural Network**: Multi-layer perceptron (100, 50 hidden units)

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## Feature Details

### Drug Features (1024 dimensions)
- **Morgan Fingerprints**: Circular fingerprints capturing molecular substructures
- **Parameters**: radius=2, nBits=1024
- **Library**: RDKit

### Protein Features (41 dimensions)
- **Amino Acid Composition**: 20 features (frequency of each amino acid)
- **CTD Composition**: 21 features based on physicochemical properties
  - Hydrophobicity, normalized van der Waals volume, polarity
  - Polarizability, charge, secondary structure, solvent accessibility

## Troubleshooting

### Common Issues

1. **Missing dataset file**
   ```
   Error: Dataset not found: data/raw/bindingdb_filtered.csv
   ```
   **Solution**: Ensure your dataset file is placed in the correct location

2. **Invalid SMILES strings**
   ```
   Warning: Failed to process X SMILES strings
   ```
   **Solution**: This is normal - invalid SMILES are replaced with zero vectors

3. **Memory issues with large datasets**
   **Solution**: Reduce dataset size or increase system RAM

4. **RDKit installation issues**
   ```cmd
   pip install rdkit-pypi
   ```

### Getting Help
- Check the log file: `dti_prediction.log`
- Verify Python version: `python --version` (should be 3.10.x)
- Verify dependencies: `pip list`

## Technical Notes

- **Random Seed**: 42 (for reproducibility)
- **Train/Test Split**: 80/20 stratified split
- **Feature Scaling**: Not applied (tree-based models handle different scales)
- **Cross-Validation**: Not implemented (single train/test split for simplicity)

## Academic Context

This application is designed for a final-year academic project focusing on:
- Bioinformatics and computational biology
- Machine learning applications in drug discovery
- Molecular descriptor analysis
- Comparative model evaluation

The implementation prioritizes clarity and educational value over computational efficiency.