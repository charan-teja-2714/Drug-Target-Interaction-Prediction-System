# Viva Questions & Answers: Drug-Target Interaction Prediction System

## Table of Contents
1. [Project Overview Questions](#1-project-overview-questions)
2. [Dataset & Data Processing](#2-dataset--data-processing)
3. [Feature Engineering](#3-feature-engineering)
4. [Machine Learning Models](#4-machine-learning-models)
5. [Implementation & Technical](#5-implementation--technical)
6. [Results & Evaluation](#6-results--evaluation)
7. [Challenges & Solutions](#7-challenges--solutions)
8. [Future Improvements](#8-future-improvements)
9. [Domain Knowledge](#9-domain-knowledge)
10. [Critical Thinking Questions](#10-critical-thinking-questions)

---

## 1. Project Overview Questions

### Q1.1: What is your project about?
**Answer**: My project is a Drug-Target Interaction (DTI) prediction system that uses machine learning to predict whether a drug molecule will interact with a protein target. It takes drug structures (as SMILES) and protein sequences as input and outputs a binary prediction (interaction or no interaction) with a confidence score.

### Q1.2: Why is DTI prediction important?
**Answer**: DTI prediction is crucial for:
- **Drug Discovery**: Identifying potential drug candidates faster and cheaper than lab experiments
- **Drug Repurposing**: Finding new uses for existing drugs
- **Side Effect Prediction**: Understanding unintended drug-protein interactions
- **Cost Reduction**: Computational screening costs thousands vs millions for wet-lab experiments

### Q1.3: What problem does your project solve?
**Answer**: Traditional experimental methods for testing drug-target interactions are expensive (thousands of dollars per test), time-consuming (weeks to months), and cannot scale to test millions of combinations. My system provides rapid computational screening in milliseconds with reasonable accuracy (79.8%).

### Q1.4: What is the scope of your project?
**Answer**: This is an academic proof-of-concept system that:
- Handles binary classification (yes/no interaction)
- Works with 20,000 balanced drug-protein pairs
- Achieves ~80% accuracy
- Provides a foundation for future research
- NOT intended for clinical deployment without extensive validation

---

## 2. Dataset & Data Processing

### Q2.1: What dataset did you use?
**Answer**: I used BindingDB, a public database containing measured binding affinities between drugs and proteins. Specifically:
- **Source**: BindingDB (https://www.bindingdb.org/)
- **Size**: 20,000 drug-protein pairs
- **Balance**: 10,000 positive + 10,000 negative interactions
- **File**: `bindingdb_balanced_20000.csv`

### Q2.2: What are the columns in your dataset?
**Answer**: Three columns:
1. **smiles**: Drug structure in SMILES notation (e.g., `CC(C)NC1=NC=NC2=C1N=CN2`)
2. **protein_sequence**: Amino acid sequence (e.g., `MTEYKLVVVGAGGVGK...`)
3. **interaction**: Binary label (0 = no interaction, 1 = interaction)

### Q2.3: How did you create the interaction labels?
**Answer**: Labels were created using binding affinity thresholds:
- Binding affinity ≤ 1000 nM → **Interaction (1)**
- Binding affinity > 1000 nM → **No Interaction (0)**

The 1000 nM threshold is a standard cutoff in pharmaceutical research representing "strong enough" binding for therapeutic effect.

### Q2.4: Why did you balance the dataset?
**Answer**: The original BindingDB has more negative samples than positive, causing class imbalance. Imbalanced data makes models biased toward the majority class. By sampling 10,000 from each class, I ensure:
- Fair learning for both classes
- Unbiased accuracy metrics
- Better generalization

### Q2.5: How did you handle missing or invalid data?
**Answer**: 
- **Missing values**: Dropped rows with missing SMILES, sequences, or labels
- **Invalid SMILES**: Converted to zero vectors (no crash)
- **Duplicates**: Removed duplicate drug-protein pairs
- **Validation**: Checked that interaction labels are only 0 or 1

### Q2.6: What is the train-test split ratio?
**Answer**: 80/20 stratified split:
- **Training**: 16,000 samples (80%)
- **Testing**: 4,000 samples (20%)
- **Stratified**: Maintains class balance in both sets (50% positive, 50% negative)

---

## 3. Feature Engineering

### Q3.1: What is SMILES notation?
**Answer**: SMILES (Simplified Molecular Input Line Entry System) is a text-based representation of chemical structures. For example:
- Aspirin: `CC(=O)Oc1ccccc1C(=O)O`
- Letters = atoms (C, N, O)
- Numbers = ring structures
- `=` = double bonds
- Parentheses = branches

### Q3.2: What are Morgan Fingerprints?
**Answer**: Morgan Fingerprints (also called Circular Fingerprints) convert molecular structures into fixed-length binary vectors. The algorithm:
1. Starts from each atom
2. Looks at neighboring atoms within a radius (I use radius=2)
3. Generates unique identifiers for local structures
4. Hashes these into a 1024-bit vector

Example: Molecule → `[0,1,0,0,1,1,0,1,...]` (1024 bits)

### Q3.3: Why did you choose Morgan Fingerprints?
**Answer**: 
- **Standard in cheminformatics**: Widely used and validated
- **Captures substructures**: Identifies functional groups important for binding
- **Fixed length**: Easy to use with ML models
- **Efficient**: Fast computation with RDKit library

### Q3.4: What are the protein features you extracted?
**Answer**: I extracted 41 protein features:
1. **Amino Acid Composition (20 features)**: Frequency of each of the 20 standard amino acids
2. **CTD Composition (21 features)**: Groups amino acids by 7 physicochemical properties (hydrophobicity, volume, polarity, polarizability, charge, secondary structure, solvent accessibility)

### Q3.5: What is Amino Acid Composition (AAC)?
**Answer**: AAC calculates the frequency of each amino acid in a protein sequence.

Example:
```
Sequence: MTEYKLVVVGAGGVGK (16 amino acids)
- M: 1/16 = 0.0625
- T: 1/16 = 0.0625
- K: 2/16 = 0.1250
- V: 4/16 = 0.2500
- G: 4/16 = 0.2500
...
```
Result: 20-dimensional vector

### Q3.6: What are CTD descriptors?
**Answer**: CTD (Composition, Transition, Distribution) groups amino acids by physicochemical properties. I use the Composition part, which groups amino acids into 7 properties × 3 subgroups = 21 features.

Example for hydrophobicity:
- Polar: RKEDQN
- Neutral: GASTPHY
- Hydrophobic: CLVIMFW

For each group, calculate: (count of amino acids in group) / (total sequence length)

### Q3.7: What is the total feature dimension?
**Answer**: 
- Drug features: 1024 (Morgan Fingerprints)
- Protein features: 41 (20 AAC + 21 CTD)
- **Total: 1065 features**

### Q3.8: Did you apply feature scaling?
**Answer**: No, I did not apply feature scaling because:
- Tree-based models (Random Forest, XGBoost) are scale-invariant
- They split on feature values, not distances
- 3 out of 5 models are tree-based
- For consistency, I kept the same features for all models

---

## 4. Machine Learning Models

### Q4.1: Which machine learning models did you use?
**Answer**: I used 5 models:
1. **Logistic Regression**: Linear baseline
2. **Random Forest**: Ensemble of 300 decision trees
3. **Support Vector Machine (SVM)**: RBF kernel
4. **XGBoost**: Gradient boosting
5. **Neural Network**: Multi-layer perceptron (256→128 neurons)

### Q4.2: Why did you choose these 5 models?
**Answer**: To compare different learning paradigms:
- **Logistic Regression**: Simple linear baseline
- **Random Forest**: Handles non-linearity, robust to noise
- **SVM**: Good with high-dimensional data
- **XGBoost**: State-of-the-art gradient boosting
- **Neural Network**: Learns complex non-linear patterns

This diversity helps identify which approach works best for DTI prediction.

### Q4.3: Which model performed best and why?
**Answer**: **Random Forest** achieved the best performance (79.8% accuracy, 88.3% ROC-AUC) because:
- Handles high-dimensional features well (1065 features)
- Robust to noisy and missing data
- Captures non-linear drug-protein interactions
- Ensemble method reduces overfitting
- Works well with mixed feature types (binary fingerprints + continuous AAC/CTD)

### Q4.4: Why did Logistic Regression perform worst?
**Answer**: Logistic Regression achieved only 66.4% accuracy because:
- It's a linear model that assumes linear relationships
- Drug-protein interactions are highly non-linear
- Cannot capture complex patterns in 1065-dimensional space
- Serves as a baseline to show the value of non-linear models

### Q4.5: What are the hyperparameters for Random Forest?
**Answer**:
```python
n_estimators=300          # 300 decision trees
max_depth=25              # Maximum tree depth
min_samples_split=5       # Minimum samples to split a node
min_samples_leaf=2        # Minimum samples in leaf node
class_weight="balanced"   # Handle class imbalance
n_jobs=-1                 # Use all CPU cores
random_state=42           # Reproducibility
```

### Q4.6: Did you perform hyperparameter tuning?
**Answer**: Yes, I manually tuned hyperparameters based on:
- Initial baseline runs
- Literature review of similar DTI prediction studies
- Iterative testing with different values
- Balancing performance vs training time

I did not use automated methods like Grid Search or Random Search due to computational constraints.

### Q4.7: Why didn't you use deep learning models like CNNs or Graph Neural Networks?
**Answer**: 
- **Dataset size**: 20,000 samples is relatively small for deep learning
- **Computational resources**: Limited to CPU, no GPU
- **Project scope**: Academic proof-of-concept focused on classical ML
- **Interpretability**: Traditional ML models are easier to explain
- **Future work**: Deep learning is a potential improvement

### Q4.8: Did you use GPU acceleration?
**Answer**: No, all models run on CPU because:
- Dataset size (20,000) is manageable on CPU
- Scikit-learn models are CPU-optimized
- Training time is acceptable (~5-10 minutes)
- No GPU hardware available
- XGBoost can use GPU but I used CPU version for consistency

---

## 5. Implementation & Technical

### Q5.1: What programming language and libraries did you use?
**Answer**:
- **Language**: Python 3.10
- **ML**: scikit-learn, XGBoost
- **Cheminformatics**: RDKit (molecular fingerprints)
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Persistence**: joblib

### Q5.2: Explain your project architecture.
**Answer**: Modular pipeline with 5 stages:
1. **Data Loader** (`data_loader.py`): Load and validate CSV
2. **Feature Extractor** (`feature_extraction.py`): Convert to numerical features
3. **Model Trainer** (`train.py`): Train 5 models
4. **Model Evaluator** (`evaluate.py`): Calculate metrics
5. **Main Pipeline** (`main.py`): Orchestrate everything

Each module is independent and reusable.

### Q5.3: How did you ensure reproducibility?
**Answer**:
- **Random seed = 42**: Set everywhere (train/test split, model initialization)
- **Fixed hyperparameters**: No randomness in model configuration
- **Deterministic split**: Same train/test split every run
- **Version pinning**: Fixed library versions in `requirements.txt`

### Q5.4: How do you save and load trained models?
**Answer**: Using joblib serialization:
```python
# Save
joblib.dump(model, "models/saved_models/random_forest_tuned.joblib")

# Load
model = joblib.load("models/saved_models/random_forest_tuned.joblib")
```

This preserves the entire model state for future predictions without retraining.

### Q5.5: How long does the pipeline take to run?
**Answer**:
- **Data loading**: ~5 seconds
- **Feature extraction**: ~2-3 minutes (20,000 samples)
- **Model training**: ~5-10 minutes (all 5 models)
- **Evaluation**: ~30 seconds
- **Total**: ~10-15 minutes

### Q5.6: Can your system make predictions on new drug-protein pairs?
**Answer**: Yes, using `predict_single.py`:
```python
# Load trained model
model = joblib.load("models/saved_models/random_forest_tuned.joblib")

# New pair
data = {"smiles": ["..."], "protein_sequence": ["..."]}
df = pd.DataFrame(data)

# Extract features and predict
X, _ = extractor.extract_features(df, training=False)
prediction = model.predict(X)[0]
probability = model.predict_proba(X)[0][1]
```

Prediction time: <100ms per pair.

---

## 6. Results & Evaluation

### Q6.1: What evaluation metrics did you use?
**Answer**: 5 metrics:
1. **Accuracy**: Overall correctness (79.8%)
2. **Precision**: Of predicted interactions, how many were correct (79.9%)
3. **Recall**: Of actual interactions, how many we found (79.8%)
4. **F1-Score**: Harmonic mean of precision and recall (79.8%)
5. **ROC-AUC**: Discriminative ability (88.3%)

### Q6.2: What is the difference between accuracy and precision?
**Answer**:
- **Accuracy**: (TP + TN) / Total → Overall correctness
- **Precision**: TP / (TP + FP) → Of predicted positives, how many are correct

Example: If model predicts 100 interactions, and 80 are correct:
- Precision = 80/100 = 80%
- Accuracy depends on total samples and true negatives

### Q6.3: What is ROC-AUC and why is it important?
**Answer**: ROC-AUC (Area Under the Receiver Operating Characteristic Curve) measures the model's ability to distinguish between classes.
- **0.5**: Random guessing
- **1.0**: Perfect classification
- **0.883**: My Random Forest model

It's important because it's threshold-independent and works well for imbalanced datasets.

### Q6.4: What are your final results?
**Answer**:

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Random Forest | 79.8% | 88.3% |
| XGBoost | 78.3% | 85.9% |
| SVM | 74.0% | 81.1% |
| Neural Network | 71.9% | 79.5% |
| Logistic Regression | 66.4% | 72.3% |

Best: Random Forest with 79.8% accuracy.

### Q6.5: Is 79.8% accuracy good enough?
**Answer**: 
- **For initial screening**: Yes, it's reasonable
- **For clinical use**: No, requires higher accuracy and extensive validation
- **Compared to random (50%)**: Significantly better
- **Compared to literature**: Comparable to similar studies (75-85% range)
- **Practical use**: Should be combined with experimental validation

### Q6.6: Did you use cross-validation?
**Answer**: No, I used a single 80/20 train-test split for simplicity. Cross-validation (e.g., 5-fold or 10-fold) would provide more robust evaluation but increases computational cost. This is a limitation I acknowledge and would implement in future work.

### Q6.7: How do you visualize the results?
**Answer**: I generate:
1. **Bar chart** (`model_comparison.png`): Compares all 5 metrics across models
2. **Confusion matrices**: Shows TP, TN, FP, FN for each model
3. **ROC curves**: Plots true positive rate vs false positive rate

---

## 7. Challenges & Solutions

### Q7.1: What challenges did you face during the project?
**Answer**:
1. **Class imbalance**: Original dataset had more negatives → Solved by balancing
2. **Invalid SMILES**: Some molecules couldn't be parsed → Replaced with zero vectors
3. **High dimensionality**: 1065 features → Used tree-based models that handle it well
4. **Computational time**: Feature extraction was slow → Optimized with numpy vectorization
5. **Model selection**: Choosing the right models → Tested multiple paradigms

### Q7.2: How did you handle invalid SMILES strings?
**Answer**: When RDKit cannot parse a SMILES string:
```python
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    # Replace with zero vector
    fingerprint = np.zeros(1024)
```

This prevents crashes and allows the pipeline to continue. Invalid molecules are essentially treated as "no information."

### Q7.3: Why didn't you use feature selection?
**Answer**: 
- Tree-based models (Random Forest, XGBoost) perform implicit feature selection
- They only split on informative features
- Explicit feature selection would add complexity
- With 1065 features and 20,000 samples, dimensionality is manageable
- Future work could explore dimensionality reduction (PCA, feature importance)

### Q7.4: How did you validate your implementation?
**Answer**:
1. **Unit testing**: Tested individual functions (feature extraction, data loading)
2. **Sanity checks**: Verified feature dimensions, label distribution
3. **Baseline comparison**: Logistic Regression as baseline
4. **Literature comparison**: Results align with similar studies
5. **Manual inspection**: Checked predictions on known drug-protein pairs

---

## 8. Future Improvements

### Q8.1: What improvements would you make?
**Answer**:
1. **Cross-validation**: 5-fold or 10-fold for robust evaluation
2. **Deep learning**: Graph Neural Networks for molecules
3. **3D structure**: Incorporate protein folding information
4. **Multi-class**: Predict binding affinity ranges, not just binary
5. **Ensemble stacking**: Combine predictions from all models
6. **Hyperparameter optimization**: Grid search or Bayesian optimization
7. **Larger dataset**: Scale to 100k+ samples
8. **GPU acceleration**: Speed up training

### Q8.2: How would you deploy this in production?
**Answer**:
1. **API development**: Flask/FastAPI REST API
2. **Model serving**: Load trained model, expose prediction endpoint
3. **Input validation**: Strict SMILES and sequence validation
4. **Batch processing**: Handle multiple predictions
5. **Monitoring**: Track prediction latency and accuracy
6. **Versioning**: Model versioning and A/B testing
7. **Security**: Authentication, rate limiting

**Note**: Current system is NOT production-ready, requires extensive validation.

### Q8.3: What other features could be added?
**Answer**:
- **Confidence intervals**: Uncertainty quantification
- **Explainability**: SHAP values to explain predictions
- **Drug similarity search**: Find similar drugs for a target
- **Batch prediction**: Upload CSV, get predictions for all pairs
- **Visualization**: 3D molecular structure viewer
- **Database integration**: Direct BindingDB API access

---

## 9. Domain Knowledge

### Q9.1: What is a drug target?
**Answer**: A drug target is a biological molecule (usually a protein) that a drug binds to in order to produce a therapeutic effect. Examples:
- **Enzymes**: Inhibit disease-causing reactions
- **Receptors**: Block or activate signaling pathways
- **Ion channels**: Regulate cellular activity

### Q9.2: What is binding affinity?
**Answer**: Binding affinity measures how strongly a drug binds to its target, typically measured in:
- **Ki** (inhibition constant)
- **Kd** (dissociation constant)
- **IC50** (half-maximal inhibitory concentration)

Lower values = stronger binding. Units: nM (nanomolar), μM (micromolar).

### Q9.3: What are the 20 standard amino acids?
**Answer**: 
- **Nonpolar**: A, V, L, I, M, F, W, P, G
- **Polar**: S, T, C, Y, N, Q
- **Positive**: K, R, H
- **Negative**: D, E

### Q9.4: What is the difference between drug discovery and drug repurposing?
**Answer**:
- **Drug Discovery**: Finding new molecules for a disease target (10-15 years, $1-2 billion)
- **Drug Repurposing**: Finding new uses for existing approved drugs (faster, cheaper, safer)

My system can help with both by predicting interactions.

### Q9.5: What is the role of bioinformatics in drug discovery?
**Answer**: Bioinformatics uses computational methods to:
- Predict drug-target interactions (my project)
- Analyze protein structures
- Identify disease biomarkers
- Optimize drug properties (ADME)
- Reduce experimental costs and time

---

## 10. Critical Thinking Questions

### Q10.1: What are the limitations of your approach?
**Answer**:
1. **Binary classification**: Cannot predict binding strength
2. **No 3D structure**: Ignores protein folding and binding pockets
3. **Sequence-based only**: Doesn't consider drug metabolism (ADME)
4. **Limited dataset**: 20,000 samples may not capture all interaction types
5. **No uncertainty**: Doesn't quantify prediction confidence intervals
6. **CPU-only**: Slower than GPU-accelerated deep learning

### Q10.2: Can your model predict side effects?
**Answer**: Indirectly, yes. If a drug interacts with unintended protein targets (off-targets), it may cause side effects. My model can screen a drug against multiple proteins to identify potential off-target interactions. However, predicting actual side effects requires additional information (drug metabolism, tissue distribution, etc.).

### Q10.3: How would you handle a new protein with no known interactions?
**Answer**: This is the "cold start" problem. My model can still make predictions because it uses:
- **Sequence-based features**: AAC and CTD work for any protein sequence
- **No need for prior interaction data**: Features are computed from structure alone

However, accuracy may be lower for proteins very different from training data.

### Q10.4: What if a drug has multiple targets?
**Answer**: My model predicts pairwise interactions (one drug, one protein). For a drug with multiple targets:
- Run predictions for each drug-protein pair separately
- Each prediction is independent
- Can identify primary target (highest confidence) and off-targets

### Q10.5: How do you ensure your model doesn't overfit?
**Answer**:
1. **Train-test split**: Evaluate on unseen data
2. **Ensemble methods**: Random Forest and XGBoost reduce overfitting
3. **Regularization**: Logistic Regression has built-in regularization
4. **Tree depth limits**: max_depth=25 prevents overly complex trees
5. **Validation**: Compare training vs test accuracy (no large gap observed)

### Q10.6: Why not use molecular docking instead?
**Answer**: Molecular docking simulates 3D binding but:
- **Computationally expensive**: Hours per drug-protein pair
- **Requires 3D structures**: Not always available
- **Complex setup**: Requires expertise

My ML approach:
- **Fast**: Milliseconds per prediction
- **Scalable**: Can screen millions of pairs
- **Sequence-based**: No 3D structure needed

Both approaches are complementary: ML for initial screening, docking for detailed analysis.

### Q10.7: How would you explain your model to a non-technical person?
**Answer**: "Imagine you have a key (drug) and a lock (protein). My system looks at the shape and properties of both and predicts whether the key will fit the lock. Instead of testing every key in a lab (expensive and slow), my computer program learns patterns from thousands of previous tests and makes educated guesses in seconds."

### Q10.8: What ethical considerations are there?
**Answer**:
1. **Misuse**: Predictions should not replace clinical trials
2. **Bias**: Model trained on existing drugs may miss novel interactions
3. **Transparency**: Users should understand model limitations
4. **Data privacy**: BindingDB is public, but patient data would require protection
5. **Responsibility**: Predictions are suggestions, not medical advice

### Q10.9: How does your work contribute to science?
**Answer**:
1. **Accelerates drug discovery**: Reduces time and cost
2. **Enables drug repurposing**: Find new uses for existing drugs
3. **Provides open-source tool**: Others can build upon it
4. **Demonstrates ML in bioinformatics**: Shows practical application
5. **Educational value**: Teaches DTI prediction methodology

### Q10.10: If you had unlimited resources, what would you do differently?
**Answer**:
1. **Larger dataset**: 1 million+ drug-protein pairs
2. **Deep learning**: Graph Neural Networks for molecules and proteins
3. **3D structure**: Incorporate AlphaFold protein structures
4. **Multi-task learning**: Predict binding affinity, not just binary
5. **GPU cluster**: Train larger models faster
6. **Experimental validation**: Test top predictions in wet lab
7. **Web platform**: User-friendly interface for researchers

---

## Quick Reference: Key Numbers

- **Dataset size**: 20,000 samples (10k positive, 10k negative)
- **Feature dimensions**: 1065 (1024 drug + 41 protein)
- **Train/test split**: 80/20 (16,000 / 4,000)
- **Number of models**: 5
- **Best accuracy**: 79.8% (Random Forest)
- **Best ROC-AUC**: 88.3% (Random Forest)
- **Training time**: ~10-15 minutes
- **Prediction time**: <100ms per pair
- **Random seed**: 42
- **Python version**: 3.10

---

## Tips for Viva Success

1. **Be honest**: If you don't know, say "I don't know, but I would research..."
2. **Show understanding**: Explain concepts in your own words
3. **Connect to real-world**: Relate to drug discovery applications
4. **Acknowledge limitations**: Shows critical thinking
5. **Discuss future work**: Shows vision and planning
6. **Stay calm**: Take a breath before answering
7. **Ask for clarification**: If question is unclear, ask to rephrase

**Good luck with your viva!**
