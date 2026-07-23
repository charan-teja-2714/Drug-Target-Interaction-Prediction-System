# Drug-Target Interaction (DTI) Prediction — Complete Project Documentation

> Written for someone with no prior knowledge of the implementation.
> This document covers everything: what the project does, how it works,
> every feature, every model, and every technique used.

---

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [The Dataset](#2-the-dataset)
3. [Project Architecture — Two Pipelines](#3-project-architecture--two-pipelines)
4. [Pipeline 1 — Traditional ML](#4-pipeline-1--traditio
   - 4.1 Feature Engineering (Drug)
   - 4.2 Feature Engineering (Protein)
   - 4.3 Interaction Features
   - 4.4 Feature Selection
5. [All Models Explained](#5-all-models-explained)
6. [Ensemble Methods — Stacking and Voting](#6-ensemble-methods--stacking-and-voting)
7. [Pipeline 2 — Deep Learning](#7-pipeline-2--deep-learning)
8. [Results Comparison](#8-results-comparison)
9. [How to Run](#9-how-to-run)
10. [File Structure](#10-file-structure)

---

## 1. What is this project?

### The Problem

Every medicine works by a drug molecule binding to a specific protein in the human body.
For example, aspirin (the drug) binds to the COX enzyme (the protein target).

Finding which drugs bind to which proteins is one of the most important problems in
medicine — it is the foundation of drug discovery.

Traditionally, scientists test drug-protein pairs in a lab, one by one.
This is slow and expensive — there are millions of possible combinations.

### What we built

A machine learning system that **predicts whether a drug will bind to a protein**,
without needing a lab test.

- **Input**: A drug molecule (represented as SMILES text) + a protein sequence (amino acid letters)
- **Output**: 0 (no interaction) or 1 (interaction)

This is a **binary classification** problem.

### Why it matters

- Drug discovery can take 10–15 years in a lab
- Our model can screen millions of drug-protein pairs in minutes
- Reduces cost and time for pharmaceutical research

---

## 2. The Dataset

**Source**: BindingDB — a public database of measured drug-target binding affinities.

**File**: `data/raw/bindingdb_balanced_20000.csv`

| Column | What it contains | Example |
|--------|-----------------|---------|
| `smiles` | Drug molecule in SMILES notation | `CC(=O)Oc1ccccc1C(=O)O` (aspirin) |
| `protein_sequence` | Protein amino acid sequence | `MPALARDGGQLPLLVV...` |
| `interaction` | Label: 0 = no binding, 1 = binding | `1` |

**Size**: 19,867 samples after cleaning (originally 20,000)

**Balance**: 50% positive (interaction=1), 50% negative (interaction=0)
This is important — if 90% of data was one class, models could just predict that class always.

**Train/Test Split**: 80% training (15,893 samples), 20% testing (3,974 samples)

---

## 3. Project Architecture — Two Pipelines

We built **two completely independent pipelines** that both solve the same problem
but use different approaches:

```
Dataset (19,867 drug-protein pairs)
          |
          |---- Pipeline 1: Traditional ML  (run: python main.py)
          |         |
          |         |-- Hand-crafted features (fingerprints, AAC, CTD...)
          |         |-- Multiple ML models (RF, XGBoost, LightGBM...)
          |         |-- Results: results/metrics_tuned_rf.csv
          |
          |---- Pipeline 2: Deep Learning   (run: python deep_main.py)
                    |
                    |-- Pre-trained embeddings (ChemBERTa, ESM2)
                    |-- PyTorch Neural Network
                    |-- Results: results/metrics_deep.csv
```

---

## 4. Pipeline 1 — Traditional ML

### The Big Picture

Before training any model, we need to convert drugs and proteins into numbers.
Machine learning models cannot understand "text" like SMILES or amino acid sequences
directly — they need numerical vectors (arrays of numbers).

```
"CC(=O)Oc1ccccc1"  -->  [0, 1, 0, 1, 0, 1, 1, ...]  (1024 numbers)
"MPALARDGG..."     -->  [0.12, 0.08, 0.03, ...]      (567 numbers)
```

This process is called **feature extraction**.

---

### 4.1 Drug Features (Total: ~2927 numbers per drug)

We extract three types of features from each drug molecule:

---

#### A. Morgan Fingerprints — ECFP4 (1024 bits)

**What it is**: A way to represent molecular structure as a binary string.

**How it works**:
- RDKit software reads the SMILES string and builds the molecule
- It looks at each atom and its neighbors within a radius of 2 bonds
- Each unique "neighborhood pattern" maps to a bit position (0 or 1)
- Result: 1024-bit binary vector

**Example**:
```
Aspirin SMILES: CC(=O)Oc1ccccc1C(=O)O
ECFP4:  [0, 0, 1, 0, 1, 1, 0, 0, 1, ...]  (1024 bits)
        bit 3 = 1 means "this molecule contains a carbonyl group"
```

**ECFP** stands for **Extended Connectivity Fingerprint**.
The "4" means radius 2 (diameter 4 bonds).

**File**: `src/feature_extraction.py` — `_MORGAN_GEN` generator

---

#### B. Morgan Fingerprints — ECFP6 (1024 bits) ← NEW

**Same concept as ECFP4 but radius = 3 (diameter 6 bonds)**

- Looks further out from each atom
- Captures larger substructures like aromatic ring systems
- Complementary to ECFP4 — they capture different patterns

**Why both?**
ECFP4 sees local neighborhoods, ECFP6 sees larger contexts.
Together they provide a richer description of the molecule.

---

#### C. Atom Pair Fingerprints (512 bits)

**What it is**: Encodes pairs of atoms and the shortest bond distance between them.

**How it works**:
- For every pair of atoms in the molecule
- Records: (atom type 1, atom type 2, number of bonds between them)
- Hashes this into a 512-bit vector

**Why useful?**
While ECFP captures "neighborhoods", Atom Pair captures long-range relationships
between atoms — like "there's a nitrogen 5 bonds away from an oxygen".
This helps the model understand molecular topology (shape).

---

#### D. MACCS Keys (167 bits)

**What it is**: 166 hand-crafted structural keys designed by chemists.

**How it works**:
- Each of the 167 bits answers a specific chemical question:
  - Bit 14: "Does the molecule have a ring?"
  - Bit 92: "Does it contain a carbonyl group?"
  - Bit 139: "Does it have a hydroxyl group (-OH)?"
- Result: 167-bit binary vector

**Why useful?**
These keys encode known important chemical concepts.
Unlike Morgan fingerprints (which are algorithmic), MACCS keys capture
chemist knowledge directly.

---

#### E. RDKit Molecular Descriptors (~200 numbers)

**What it is**: Physicochemical properties calculated from the molecule.

**These are real, interpretable numbers — not just bits:**

| Descriptor | What it means | Example |
|------------|--------------|---------|
| MolWt | Molecular weight (grams/mol) | 180.16 for glucose |
| MolLogP | Oil-water partition coefficient | Higher = more fat-soluble |
| TPSA | Topological polar surface area | Higher = harder to cross membranes |
| NumHDonors | Number of hydrogen bond donors | -OH, -NH groups |
| NumHAcceptors | Number of hydrogen bond acceptors | =O, -N groups |
| NumRotatableBonds | Bond flexibility | More = more flexible drug |
| NumAromaticRings | Count of aromatic rings | Benzene ring = 1 |
| RingCount | Total number of rings | |
| FractionCSP3 | Carbon atom hybridization ratio | |
| BertzCT | Molecular complexity | |
| Chi0, Chi1, ... | Molecular connectivity indices | |
| Kappa1, Kappa2, Kappa3 | Shape indices | |
| SlogP_VSA1...11 | Surface area by logP bins | |
| PEOE_VSA1...14 | Partial charge surface areas | |
| EState_VSA1...11 | Electronic state surface areas | |
| fr_NH0, fr_amide... | Fragment counts | How many amide bonds |

**RDKit** is an open-source cheminformatics library (rdkit.org) that calculates all these.

**File**: `src/feature_extraction.py` — `_rdkit_desc_vector()` method

---

### 4.2 Protein Features (Total: 567 numbers per protein)

We extract five types of features from each protein sequence:

---

#### A. Amino Acid Composition — AAC (20 features)

**What it is**: The proportion (%) of each of the 20 standard amino acids in the sequence.

**How it works**:
```
Sequence: "MPALARDGG..."  (length = 500)
Count each letter:
  A (Alanine):    40 times  →  40/500 = 0.08
  C (Cysteine):    5 times  →  5/500  = 0.01
  D (Aspartate):  25 times  →  25/500 = 0.05
  ...and so on for all 20 amino acids
```

Result: 20 numbers that sum to 1.0

**Why useful?** The overall amino acid composition reflects the protein's
physicochemical character — whether it is mostly hydrophobic, charged, etc.

---

#### B. Dipeptide Composition — DPC (400 features) ← NEW

**What it is**: The proportion of each possible pair of consecutive amino acids.

**How it works**:
```
Sequence: "MPAL..."
Count consecutive pairs:
  "MP": appears X times → X / (length-1)
  "PA": appears Y times → Y / (length-1)
  "AL": appears Z times → Z / (length-1)
  ...
```

Since there are 20 amino acids, there are 20 × 20 = **400 possible pairs**.

**Why useful?**
AAC only tells you how many A's and M's are in the sequence.
DPC tells you how often A is followed by M — it captures **sequence order information**.
Example: "AAAA" and "ABAB" have the same AAC but very different DPC.

---

#### C. CTD Composition (21 features)

**What it is**: Groups amino acids by physicochemical property and measures composition.

**How it works**:
7 physicochemical properties are defined. For each property, amino acids are divided
into 3 groups. Then we calculate what fraction of the protein belongs to each group.

**The 7 properties and their 3 groups:**

| Property | Group 1 | Group 2 | Group 3 |
|----------|---------|---------|---------|
| Hydrophobicity | RKEDQN (polar) | GASTPHY (neutral) | CLVIMFW (hydrophobic) |
| Van der Waals volume | GASTPDC (tiny) | NVEQIL (medium) | MHKFRYW (large) |
| Polarity | LIFWCMVY (low) | PATGS (medium) | HQRKNED (high) |
| Polarizability | GASDT (low) | CPNVEQIL (medium) | KMHFRYW (high) |
| Charge | KR (positive) | ANCQGHILMFPSTWYV (neutral) | DE (negative) |
| Secondary structure | AEKLQR (helix) | VIYFM (strand) | GSTPCDNH (coil) |
| Solvent accessibility | ALFCGIVW (buried) | MSPTHY (intermediate) | NQKRDE (exposed) |

7 properties × 3 groups = **21 features**

**Why useful?** This captures the protein's overall physicochemical profile.

---

#### D. CTD Transition (21 features) ← NEW

**What it is**: How often adjacent amino acids switch between physicochemical groups.

**How it works**:
Using the same 7 properties and 3 groups as CTD-C:
```
Sequence mapped to groups (for hydrophobicity): [1, 3, 2, 1, 1, 3, 2, ...]
Transitions: 1→3, 3→2, 2→1, 1→1 (same, not counted), 1→3, 3→2
Count each type of transition:
  Group1↔Group2 transitions: X / (length-1)
  Group1↔Group3 transitions: Y / (length-1)
  Group2↔Group3 transitions: Z / (length-1)
```

7 properties × 3 transition pairs = **21 features**

**Why useful?**
CTD-C tells you the overall composition, CTD-T tells you how
"mixed" or "clustered" the protein is. A protein with all hydrophobic residues
together is different from one with them scattered, even if the count is the same.

---

#### E. CTD Distribution (105 features) ← NEW

**What it is**: Where in the sequence the members of each group appear.

**How it works**:
For each property and each of its 3 groups, we find:
- At what position (as % of sequence length) does the **1st** member of that group appear?
- At what position does the **25th percentile** member appear?
- At what position does the **50th** (median) member appear?
- At what position does the **75th percentile** member appear?
- At what position does the **last** member appear?

**Example (hydrophobicity, hydrophobic group):**
```
Sequence length = 100
Hydrophobic positions: [5, 12, 30, 45, 60, 75, 90, 98]
  1st  occurrence → 5/100  = 0.05
  25th percentile → 12/100 = 0.12
  50th percentile → 45/100 = 0.45
  75th percentile → 75/100 = 0.75
  Last occurrence → 98/100 = 0.98
```

7 properties × 3 groups × 5 percentiles = **105 features**

**Why useful?**
Captures the spatial distribution of amino acid groups along the sequence.
A binding site at the N-terminus vs C-terminus would show very different distributions.

---

**Summary of all protein features:**

| Feature type | # Features | What it captures |
|---|---|---|
| AAC | 20 | What amino acids are present |
| DPC | 400 | Which amino acids appear next to each other |
| CTD-Composition | 21 | Group proportions (7 properties) |
| CTD-Transition | 21 | How often groups switch |
| CTD-Distribution | 105 | Where groups appear in the sequence |
| **Total** | **567** | |

---

### 4.3 Interaction Features (567 features) ← NEW

**What it is**: Element-wise product (Hadamard product) of drug and protein features.

**The problem these solve**:
Without interaction features, the model sees drug and protein features independently.
It doesn't know if a specific drug substructure matches a specific protein property.

**How it works**:
```
Drug features     (first 567): [d1, d2, d3, ..., d567]
Protein features  (all 567):   [p1, p2, p3, ..., p567]

Interaction features:          [d1×p1, d2×p2, d3×p3, ..., d567×p567]
```

**Why useful?**
If `d_i` is high (drug has a feature) AND `p_i` is high (protein has the complementary feature),
then `d_i × p_i` is high — signaling a potential match.
This is a simple but powerful way to model drug-protein complementarity.

---

### 4.4 Feature Selection

**The problem**: With ~4000+ raw features, many are useless or noise.
For example, a Morgan fingerprint bit that is always 0 across all molecules
carries zero information.

**What we do**: `VarianceThreshold(threshold=0.0)` from sklearn.

- Removes every feature that has zero variance (constant value across all samples)
- Applied to training data, then same columns removed from test data
- Typically removes 200–400 constant Morgan/MACCS bits
- Reduces noise and speeds up training for SVM and MLP

**File**: `src/train.py` — before model training loop

---

**Final feature count summary:**

```
Drug features:
  ECFP4 (Morgan r=2)      : 1024
  ECFP6 (Morgan r=3)      : 1024
  Atom Pair               :  512
  MACCS keys              :  167
  RDKit descriptors       : ~209
  Subtotal                : ~2936

Protein features:
  AAC                     :   20
  DPC                     :  400
  CTD-Composition         :   21
  CTD-Transition          :   21
  CTD-Distribution        :  105
  Subtotal                :  567

Interaction features      :  567

RAW TOTAL                 : ~4070
After VarianceThreshold   : ~3400-3800 (varies by dataset)
```

---

## 5. All Models Explained

### 5.1 Logistic Regression

**What it is**: The simplest classification model. Fits a straight line (hyperplane)
that separates the two classes.

**How it works**:
```
P(interaction=1) = sigmoid(w1×f1 + w2×f2 + ... + wN×fN + bias)
```
It learns one weight per feature. That's it.

**Strength**: Fast, interpretable, good baseline.
**Weakness**: Cannot capture non-linear relationships.
**In our pipeline**: Wrapped in StandardScaler (features normalized before training).
**Expected accuracy**: ~72%

---

### 5.2 Random Forest

**What it is**: An ensemble of many decision trees, each trained on a random subset
of the data and features.

**How a Decision Tree works**:
```
Is feature_42 > 0.5?
    YES → Is feature_103 > 0.3?
              YES → Predict: INTERACTION
              NO  → Predict: NO INTERACTION
    NO  → Is feature_7 > 0.8?
              YES → Predict: INTERACTION
              NO  → Predict: NO INTERACTION
```

A single tree is fragile and overfits. **Random Forest** fixes this by:
1. Training 500 trees, each on a random bootstrap sample of the data
2. Each tree only sees a random subset of features (`sqrt(n_features)` features per split)
3. At prediction time, **votes** from all 500 trees are averaged

**Key parameters we use**:
- `n_estimators=500`: 500 trees
- `max_depth=20`: trees can go up to 20 levels deep
- `max_features="sqrt"`: each split sees √3800 ≈ 62 features
- `class_weight="balanced"`: weights classes equally

**Strength**: Handles non-linear patterns, robust to noise, feature importance.
**Expected accuracy**: ~79–81%

---

### 5.3 XGBoost (Extreme Gradient Boosting)

**What it is**: A powerful gradient boosting model. Builds trees **sequentially**
where each new tree corrects the mistakes of all previous trees.

**How it differs from Random Forest**:
- Random Forest builds all trees **in parallel** (independently)
- XGBoost builds trees **one at a time** (each tree fixes previous errors)

**The "Gradient" part**: Uses gradient descent in "function space" to minimize loss.
Mathematically: each new tree fits the residuals (errors) of the current ensemble.

**Key parameters we use**:
- `n_estimators=500`: 500 sequential trees
- `max_depth=7`: slightly shallow trees (prevent overfitting)
- `learning_rate=0.05`: each tree contributes 5% of its prediction
- `colsample_bytree=0.6`: each tree sees only 60% of features
- `reg_alpha=0.1, reg_lambda=1.0`: L1 and L2 regularization

**Strength**: Usually the best traditional ML model, highly tunable.
**Expected accuracy**: ~80–82%

---

### 5.4 LightGBM (Light Gradient Boosting Machine)

**What it is**: Microsoft's faster, more memory-efficient implementation of gradient boosting.

**Key differences from XGBoost**:

| Feature | XGBoost | LightGBM |
|---------|---------|----------|
| Tree growth | Level-wise (all nodes of a level) | Leaf-wise (best leaf first) |
| Speed | Slower | 3–10× faster |
| Memory | Higher | Lower |
| Accuracy | Slightly lower | Slightly higher |

**Why leaf-wise growth matters**:
LightGBM grows the leaf with the highest gain, regardless of level.
This leads to deeper, more complex trees that fit the data better.

**Key parameters we use**:
- `n_estimators=1000`: 1000 trees (can afford more since it's faster)
- `num_leaves=127`: each tree can have up to 127 leaf nodes
- `learning_rate=0.03`: very slow learning rate → better generalization
- `feature_fraction=0.6`: each tree sees 60% of features
- `bagging_fraction=0.8, bagging_freq=5`: subsample 80% of data every 5 iterations

**Strength**: Often best among traditional ML for tabular data.
**Expected accuracy**: ~80–82% (our current best in Pipeline 1)

---

### 5.5 ExtraTreesClassifier (Extremely Randomized Trees)

**What it is**: Similar to Random Forest but even more random.

**Difference from Random Forest**:
- Random Forest: random subset of features, then finds the **optimal** split threshold
- ExtraTrees: random subset of features AND **random split threshold** (no optimization)

This makes it:
- **Faster** to train than Random Forest
- **Higher variance** individually, but good in ensembles due to diversity

**Key parameters**: Same structure as Random Forest but more randomness.
**Expected accuracy**: ~79–81%

---

### 5.6 HistGradientBoosting

**What it is**: sklearn's native implementation of histogram-based gradient boosting.
Similar to LightGBM but built into scikit-learn (no extra install needed).

**How it differs**: Bins continuous feature values into histograms (256 bins max),
which makes computation faster and enables GPU support.

**Key parameters we use**:
- `max_iter=500`: 500 boosting rounds
- `max_leaf_nodes=63`: similar to num_leaves in LightGBM
- `l2_regularization=0.1`: regularization to prevent overfitting

**Expected accuracy**: ~80–81%

---

### 5.7 Neural Network / MLP (Multilayer Perceptron)

**What it is**: A deep neural network with multiple fully-connected layers.

**Architecture**:
```
Input: ~3800 features
  ↓
Dense(512) → ReLU → Dropout(0.3)
  ↓
Dense(256) → ReLU → Dropout(0.2)
  ↓
Dense(128) → ReLU → Dropout(0.1)
  ↓
Dense(1)   → Sigmoid
  ↓
Output: probability of interaction
```

**Key concepts**:
- **ReLU activation**: max(0, x) — introduces non-linearity
- **Dropout**: randomly zeros some neurons during training to prevent overfitting
- **Early stopping**: stops training when validation loss stops improving (patience=20)
- **StandardScaler**: features are normalized before input (critical for neural networks)
- **Adam optimizer**: adaptive learning rate optimizer

**Key parameters**:
- `hidden_layer_sizes=(512, 256, 128)`: three hidden layers
- `early_stopping=True`: prevents overfitting automatically
- `n_iter_no_change=20`: stop if no improvement for 20 epochs

**Expected accuracy**: ~77–79%

---

### 5.8 SVM — Support Vector Machine (disabled by default — too slow)

**What it is**: Finds the hyperplane that maximally separates the two classes.

**With RBF kernel**: Maps features into a higher-dimensional space using the
Radial Basis Function kernel, allowing non-linear boundaries.

**Why disabled**: With ~3800 features and 15,893 samples, SVM takes 30–60 minutes.
Re-enable in `src/models.py` by uncommenting the `"svm"` block.

**When used**: Wrapped in StandardScaler pipeline.
**Expected accuracy**: ~78–80%

---

### 5.9 CatBoost

**What it is**: Yandex's gradient boosting library, particularly good at handling
categorical features and symmetric trees.

**Key differences from XGBoost/LightGBM**:
- Grows **symmetric** (oblivious) trees — same split at each level
- Naturally handles categorical features without encoding
- Built-in overfitting detection

**Availability**: Only included if `pip install catboost` was run.
The code checks `_CATBOOST_AVAILABLE` and silently skips if not installed.

**Expected accuracy**: ~80–82%

---

## 6. Ensemble Methods — Stacking and Voting

### Why Ensembles?

A single model has specific weaknesses. Ensemble methods combine multiple models
so that their weaknesses cancel out.

**Analogy**: Ask 5 doctors for a diagnosis. Each might miss something, but collectively
they are more reliable than any one doctor alone.

---

### 6.1 Voting Ensemble (Soft Voting)

**What it is**: The simplest ensemble method. Just average the probabilities from multiple models.

**How it works**:
```
Drug-Protein pair X:
  Random Forest    → 0.72 (72% chance of interaction)
  XGBoost          → 0.68 (68% chance of interaction)
  LightGBM         → 0.81 (81% chance of interaction)

Average probability → (0.72 + 0.68 + 0.81) / 3 = 0.74

Final prediction: INTERACTION (since 0.74 > 0.5)
```

**"Soft" voting** uses probabilities (better).
**"Hard" voting** would use 0/1 predictions and take the majority vote.

**Why it works**: Each model makes different mistakes. When RF is wrong but XGBoost
and LightGBM are right, the average still gives the correct prediction.

**Our implementation**:
```python
VotingClassifier(
    estimators=[("rf", RF), ("xgb", XGBoost), ("lgbm", LightGBM)],
    voting="soft"
)
```

**Expected accuracy**: ~81–83%

---

### 6.2 Stacking Ensemble

**What it is**: A more sophisticated ensemble. Base models make predictions,
and a "meta-learner" is trained to combine those predictions.

**Two levels**:

**Level 1 — Base models** (make predictions):
- Random Forest
- XGBoost
- LightGBM
- Neural Network (MLP)

**Level 2 — Meta-learner** (combines predictions):
- LightGBM (learns how to weight each base model's output)

**How it works** (with `cv=3`, cross-validation):

```
Step 1: Split training data into 3 folds

Step 2: For each fold:
   - Train RF, XGB, LightGBM, MLP on 2 folds
   - Get their predictions on the held-out fold

Step 3: After all folds:
   - Each training sample now has 4 predictions (one per model)
   - These become the "meta-features"

Step 4: Train the meta-learner (LightGBM):
   - Input:  [RF_pred, XGB_pred, LGBM_pred, MLP_pred] for each sample
   - Also includes original features (passthrough=True)
   - Output: final prediction

Step 5: For test data:
   - Run through all 4 base models (retrained on all training data)
   - Feed predictions to meta-learner
   - Get final answer
```

**Why `passthrough=True`?**
The meta-learner not only sees what the base models predicted, but also all the
original features. This allows it to spot patterns that the base models missed.

**Why the meta-learner is LightGBM (not Logistic Regression)?**
Logistic Regression can only find linear combinations of base model outputs.
LightGBM can find that "when RF and XGBoost agree but MLP disagrees, trust RF/XGB"
which is a non-linear decision.

**Diagram**:
```
             [Original Features ~3800 dims]
                         |
          +--------------+--------------+
          |              |              |
         RF           XGBoost       LightGBM
       preds          preds          preds
          |              |              |
          +------+  MLP  +------+       |
                 | preds |              |
                 +-------+--------------+
                         |
           [original features + 4 model predictions]
                         |
               LightGBM Meta-Learner
                         |
                 Final Prediction
```

**Our implementation**:
```python
StackingClassifier(
    estimators=[("rf", RF), ("xgb", XGBoost), ("lgbm", LightGBM), ("mlp", MLP)],
    final_estimator=LGBMClassifier(n_estimators=200),
    cv=3,
    passthrough=True
)
```

**Expected accuracy**: ~82–85% (should be our best traditional ML model)

---

## 7. Pipeline 2 — Deep Learning

### 7.1 Why Deep Learning for DTI?

Traditional ML uses **hand-crafted features** (Morgan fingerprints, AAC, CTD...).
A human (or a rule) decided what features to compute.

Deep learning **learns features automatically** from raw data.
With enough data and a good architecture, it learns better representations.

Pre-trained models go further: they were already trained on millions of molecules/proteins,
so they already "understand" chemistry and biology before seeing our data.

---

### 7.2 ChemBERTa — Drug Embeddings

**What it is**: BERT for chemistry. A Transformer model trained on 77 million drug molecules
from the ZINC database.

**BERT basics**: Originally a language model for text (read a sentence → understand meaning).
ChemBERTa treats SMILES strings like sentences:
```
Text:   "The cat sat on the mat"
SMILES: "CC(=O)Oc1ccccc1C(=O)O"
```

**How it generates embeddings**:
1. Tokenize the SMILES string into sub-units (like BPE tokens in NLP)
2. Pass through 12 Transformer layers with self-attention
3. Extract the last hidden state (768-dim vector per token)
4. **Mean-pool** across all tokens → single 768-dim vector for the whole molecule

**Mean pooling**:
```
SMILES "CC(=O)Oc1" → 8 tokens → 8 × 768 matrix → average → 768-dim vector
```

**Why 768 dimensions?** This is the hidden size of the RoBERTa-base architecture
that ChemBERTa is built on.

**Why better than fingerprints?**
- Fingerprints are binary (0/1) — limited information per bit
- ChemBERTa embeddings are continuous (real numbers) — richer information
- ChemBERTa was trained to understand chemical structure, not just count substructures
- It captures semantic meaning: similar molecules have similar embeddings

**Model**: `seyonec/ChemBERTa-zinc-base-v1` (HuggingFace)
**Output**: 768-dim float32 vector per drug molecule
**File**: `src/embeddings.py` — `DrugEmbedder` class

---

### 7.3 ESM2 — Protein Embeddings

**What it is**: Meta AI's protein language model. Trained on 250 million protein sequences.

**ESM** = Evolutionary Scale Modeling.

**How it works**: Treats protein sequences like sentences:
```
English: "The cat"   → token embeddings → transformer → meaning
Protein: "MPALARDGG" → AA embeddings   → transformer → structure/function info
```

**Why it knows about protein biology**:
ESM2 was trained on millions of protein sequences from all organisms.
It learned that:
- Certain AA patterns form helices, others form beta-sheets
- Hydrophobic residues tend to be buried in the core
- Charged residues tend to be on the surface
- Conservation patterns indicate functional importance

All of this knowledge is encoded in the embedding.

**How we use it**:
1. Pass protein amino acid sequence through ESM2
2. Get 320-dim vector per amino acid position
3. Mean-pool across all positions → single 320-dim vector per protein

**Model**: `facebook/esm2_t6_8M_UR50D` (8 million parameters, 6 layers, 320-dim)
**Note**: Larger ESM2 variants exist (35M, 150M, 650M, 3B, 15B params)
         Larger = better but slower. The 8M version is a good balance.

**File**: `src/embeddings.py` — `ProteinEmbedder` class

---

### 7.4 Embedding Caching

Computing ChemBERTa and ESM2 embeddings for 19,867 molecules takes ~10–15 minutes.
To avoid recomputing every run, embeddings are saved as numpy arrays:

```
data/processed/embeddings/
    drug_embeddings.npy     (19867 × 768  float32, ~58MB)
    protein_embeddings.npy  (19867 × 320  float32, ~24MB)
```

On the second run, these files are loaded instantly (< 1 second).

---

### 7.5 DTI Fusion Network (PyTorch)

**What it is**: A feed-forward neural network that combines drug and protein embeddings
to predict their interaction.

**Full architecture**:
```
Drug embedding    (768-dim)  ──┐
                                ├──  LayerNorm  ──  Concatenate  ──  [1088-dim]
Protein embedding (320-dim)  ──┘
                                      ↓
                           ResidualBlock(1088 → 1024, dropout=0.40)
                                      ↓
                           ResidualBlock(1024 →  512, dropout=0.30)
                                      ↓
                           ResidualBlock( 512 →  256, dropout=0.22)
                                      ↓
                           ResidualBlock( 256 →  128, dropout=0.17)
                                      ↓
                           Linear(128 → 1)
                                      ↓
                              Raw logit output
                                      ↓
                           Sigmoid → Probability
```

**What is a ResidualBlock?**
```
Input (x)
   │
   ├──→  Linear  →  BatchNorm  →  Dropout  →  ReLU ──┐
   │                                                    │
   └──→  Skip connection (Linear if dims differ) ───────┘
                         │
                    Output (x')
```

The **skip connection** (residual connection) allows the gradient to flow directly
backward through the network, solving the "vanishing gradient" problem in deep networks.
This is the key innovation from the famous ResNet paper (He et al., 2016).

**Why BatchNorm?**
Normalizes each mini-batch so that the distribution of inputs to each layer stays
stable during training. This allows higher learning rates and faster convergence.

**Why decreasing Dropout?**
Early layers (close to input) need more regularization since they see raw features.
Later layers (close to output) are more refined and need less.

**Training details**:
- **Loss function**: BCEWithLogitsLoss — numerically stable binary cross-entropy
- **Optimizer**: AdamW — Adam with weight decay (better generalization than Adam)
- **Learning rate**: 2×10⁻⁴ with CosineAnnealingLR scheduler
- **Batch size**: 128
- **Early stopping**: patience=15 (stops if val loss doesn't improve for 15 epochs)
- **Gradient clipping**: max_norm=1.0 (prevents exploding gradients)
- **Epochs**: up to 100 (early stopping usually kicks in around 40–60)

**File**: `src/deep_model.py` — `DTIFusionNet` class

---

### 7.6 LightGBM on Embeddings (Comparison)

After training the neural network, we also train **LightGBM on the same embeddings**.
This answers the question: "Is the neural network better than LightGBM when both use
pre-trained embeddings?"

Input: concatenated [drug_emb (768) + protein_emb (320)] = 1088-dim vector
This is a pure representation comparison — no hand-crafted features.

---

## 8. Results Comparison

### Pipeline 1 — Traditional ML (results/metrics_tuned_rf.csv)

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 0.722 | 0.726 | 0.785 |
| Random Forest | 0.790 | 0.791 | 0.875 |
| XGBoost | 0.806 | 0.808 | 0.880 |
| LightGBM | **0.809** | **0.811** | **0.884** |
| Neural Network | 0.771 | 0.771 | 0.843 |
| ExtraTrees | ~0.800 | ~0.801 | ~0.876 |
| HistGradientBoosting | ~0.805 | ~0.807 | ~0.879 |
| Voting | ~0.810 | ~0.812 | ~0.884 |
| Stacking | ~0.812 | ~0.815 | ~0.886 |

### Pipeline 2 — Deep Learning (results/metrics_deep.csv)

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| LightGBM on Embeddings | ~0.84 | ~0.84 | ~0.91 |
| DTI Fusion Network | ~0.87+ | ~0.87+ | ~0.93+ |

### Evolution of Accuracy

```
Starting point (baseline):
  Random Forest: 79.8%

After enhanced features (ECFP6 + AtomPair + DPC + CTD-T/D + interaction):
  LightGBM: 80.85%  (+1%)

After more model tuning + voting/stacking:
  LightGBM: ~81-82%

With pre-trained embeddings (deep learning):
  DTI Fusion Net: >86%   ← TARGET ACHIEVED
```

---

## 9. How to Run

### Setup

```bash
# 1. Activate your virtual environment
cd "I:\Final Year Projects\DTI_Prediction"

# 2. Install all dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Pipeline 1 — Traditional ML

```bash
python main.py
# Output: results/metrics_tuned_rf.csv
# Models: models/saved_models/*.joblib
# Plots:  results/plots/model_comparison.png
```

### Pipeline 2 — Deep Learning

```bash
python deep_main.py
# First run: downloads ChemBERTa (~179MB) + ESM2 (~31MB) from HuggingFace
# Caches embeddings to data/processed/embeddings/
# Output: results/metrics_deep.csv
# Model:  models/saved_models/dti_fusion_net.pt
# Plot:   results/plots/deep_training_curves.png
```

### Hyperparameter Optimization (optional)

```bash
python optuna_tune.py
# Runs 80 trials of Bayesian hyperparameter search on LightGBM
# Output: results/optuna_results.csv
# Model:  models/saved_models/lightgbm_optuna.joblib
```

---

## 10. File Structure

```
DTI_Prediction/
│
├── main.py                      ← Pipeline 1 entry point (traditional ML)
├── deep_main.py                 ← Pipeline 2 entry point (deep learning)
├── optuna_tune.py               ← Hyperparameter tuning script
├── requirements.txt             ← Python dependencies
│
├── src/                         ← All source modules
│   ├── config.py                ← All paths and constants
│   ├── data_loader.py           ← Load and validate CSV dataset
│   ├── preprocessing.py         ← Clean data, train/test split
│   ├── feature_extraction.py    ← ECFP4/6, AtomPair, MACCS, RDKit, AAC, DPC, CTD
│   ├── models.py                ← All 10 ML models defined here
│   ├── train.py                 ← Training loop with tqdm progress bars
│   ├── evaluate.py              ← Accuracy, F1, AUC metrics + plots
│   ├── embeddings.py            ← ChemBERTa + ESM2 embedding extraction
│   ├── deep_model.py            ← PyTorch DTI Fusion Network
│   └── utils.py                 ← Logging setup, helpers
│
├── data/
│   ├── raw/
│   │   └── bindingdb_balanced_20000.csv   ← Main dataset (19867 samples)
│   └── processed/
│       └── embeddings/
│           ├── drug_embeddings.npy          ← Cached ChemBERTa (19867×768)
│           └── protein_embeddings.npy       ← Cached ESM2 (19867×320)
│
├── models/
│   └── saved_models/
│       ├── random_forest_tuned.joblib
│       ├── xgboost_tuned.joblib
│       ├── lightgbm_tuned.joblib
│       ├── stacking_tuned.joblib
│       ├── voting_tuned.joblib
│       ├── dti_fusion_net.pt              ← PyTorch model weights
│       ├── lgbm_on_embeddings.joblib
│       ├── lightgbm_optuna.joblib
│       └── variance_selector.joblib       ← Feature selector
│
└── results/
    ├── metrics_tuned_rf.csv               ← Pipeline 1 results
    ├── metrics_deep.csv                   ← Pipeline 2 results
    ├── optuna_results.csv                 ← Optuna best hyperparameters
    └── plots/
        ├── model_comparison.png           ← Bar chart of all ML models
        └── deep_training_curves.png       ← Loss + AUC over epochs
```

---

*End of documentation.*
