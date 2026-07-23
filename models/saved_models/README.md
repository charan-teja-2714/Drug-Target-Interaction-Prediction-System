# Models Directory

## Included in Git (Small Models)

These lightweight models are included in the repository:

- ✅ `variance_selector.joblib` (33 KB) - Feature selector (REQUIRED)
- ✅ `logistic_regression.joblib` (9 KB) - Baseline model
- ✅ `xgboost.joblib` (358 KB) - Gradient boosting model

## Excluded from Git (Large Models)

These models exceed GitHub's 100MB limit and must be trained locally:

- ❌ Random Forest models (71-181 MB)
- ❌ SVM models (110-260 MB)
- ❌ Stacking/Voting ensembles (138-166 MB)
- ❌ Neural Network tuned (27 MB)
- ❌ Extra Trees (181 MB)

## How to Generate Missing Models

Run the training pipeline to generate all models:

```bash
# Activate environment
dti_env\Scripts\activate

# Train all models
python main.py
```

This will create all model files in `models/saved_models/`.

## Model Sizes Reference

| Model | Size | Included |
|-------|------|----------|
| variance_selector | 33 KB | ✅ |
| logistic_regression | 9 KB | ✅ |
| xgboost | 358 KB | ✅ |
| catboost_tuned | 3.1 MB | ❌ |
| hist_gradient_boosting_tuned | 4.1 MB | ❌ |
| lightgbm_tuned | 6.5 MB | ❌ |
| neural_network_tuned | 27 MB | ❌ |
| random_forest_tuned | 129 MB | ❌ |
| extra_trees_tuned | 181 MB | ❌ |
| stacking_tuned | 166 MB | ❌ |
| voting_tuned | 138 MB | ❌ |
| svm_tuned | 260 MB | ❌ |

## Alternative: Model Hosting

For sharing large models, consider:
- **Git LFS** (Large File Storage)
- **Google Drive / Dropbox** with download script
- **HuggingFace Model Hub**
- **AWS S3 / Azure Blob Storage**
