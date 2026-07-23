"""
Streamlit Frontend for DTI Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.feature_extraction import FeatureExtractor
from src.config import SAVED_MODELS_DIR, RESULTS_DIR

st.set_page_config(page_title="DTI Prediction", page_icon="💊", layout="wide")

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_dir = Path(SAVED_MODELS_DIR)
    if model_dir.exists():
        for model_file in model_dir.glob("*.joblib"):
            model_name = model_file.stem
            models[model_name] = joblib.load(model_file)
    return models

# Load feature selector
@st.cache_resource
def load_feature_selector():
    selector_path = SAVED_MODELS_DIR / "variance_selector.joblib"
    if selector_path.exists():
        return joblib.load(selector_path)
    return None

# Load metrics
@st.cache_data
def load_metrics():
    metrics_file = RESULTS_DIR / "metrics_tuned_rf.csv"
    if metrics_file.exists():
        return pd.read_csv(metrics_file)
    return None

st.title("💊 Drug-Target Interaction Prediction")
st.markdown("Predict interactions between drugs and protein targets using machine learning")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "ℹ️ About"])

# Tab 1: Prediction
with tab1:
    st.header("Make Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        smiles = st.text_area(
            "Drug SMILES",
            placeholder="Enter SMILES notation (e.g., CC(=O)OC1=CC=CC=C1C(=O)O)",
            height=100
        )
    
    with col2:
        protein_seq = st.text_area(
            "Protein Sequence",
            placeholder="Enter protein FASTA sequence (e.g., MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL)",
            height=100
        )
    
    models = load_models()
    selector = load_feature_selector()
    
    if models:
        model_name = st.selectbox("Select Model", list(models.keys()))
        
        if st.button("🔍 Predict Interaction", type="primary"):
            if not smiles or not protein_seq:
                st.error("Please enter both SMILES and protein sequence")
            else:
                with st.spinner("Extracting features and predicting..."):
                    try:
                        # Create dataframe
                        df = pd.DataFrame({
                            'smiles': [smiles],
                            'protein_sequence': [protein_seq],
                            'interaction': [0]  # dummy
                        })
                        
                        # Extract features
                        extractor = FeatureExtractor()
                        X, _ = extractor.extract_features(df, training=False)
                        
                        # Apply feature selection
                        if selector is not None:
                            X = selector.transform(X)
                        
                        # Predict
                        model = models[model_name]
                        prediction = model.predict(X)[0]
                        proba = model.predict_proba(X)[0]
                        
                        # Display results
                        st.success("✅ Prediction Complete!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Prediction", "Interaction" if prediction == 1 else "No Interaction")
                        with col2:
                            st.metric("Confidence (No Interaction)", f"{proba[0]:.2%}")
                        with col3:
                            st.metric("Confidence (Interaction)", f"{proba[1]:.2%}")
                        
                        # Interpretation guide
                        if proba[1] > 0.4:
                            st.info("💡 **Interpretation**: Moderate-to-high interaction probability. Consider experimental validation.")
                        elif proba[1] > 0.25:
                            st.warning("⚠️ **Interpretation**: Uncertain prediction. Model shows weak interaction signal.")
                        else:
                            st.info("ℹ️ **Interpretation**: Low interaction probability based on training data patterns.")
                        
                        # Visualization
                        st.subheader("Probability Distribution")
                        prob_df = pd.DataFrame({
                            'Class': ['No Interaction', 'Interaction'],
                            'Probability': proba
                        })
                        st.bar_chart(prob_df.set_index('Class'))
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    else:
        st.warning("⚠️ No trained models found. Please run `python main.py` first to train models.")

# Tab 2: Model Performance
with tab2:
    st.header("Model Performance Metrics")
    
    metrics_df = load_metrics()
    
    if metrics_df is not None:
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=metrics_df.columns[1:]), use_container_width=True)
        
        st.subheader("Performance Comparison")
        
        metric_to_plot = st.selectbox("Select Metric", ["accuracy", "precision", "recall", "f1_score", "roc_auc"])
        
        chart_data = metrics_df.set_index('model')[metric_to_plot].sort_values(ascending=False)
        st.bar_chart(chart_data)
        
        # Best models
        st.subheader("🏆 Best Models")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_acc = metrics_df.loc[metrics_df['accuracy'].idxmax()]
            st.metric("Best Accuracy", f"{best_acc['accuracy']:.4f}", best_acc['model'])
        
        with col2:
            best_f1 = metrics_df.loc[metrics_df['f1_score'].idxmax()]
            st.metric("Best F1-Score", f"{best_f1['f1_score']:.4f}", best_f1['model'])
        
        with col3:
            best_auc = metrics_df.loc[metrics_df['roc_auc'].idxmax()]
            st.metric("Best ROC-AUC", f"{best_auc['roc_auc']:.4f}", best_auc['model'])
    else:
        st.warning("⚠️ No metrics found. Please run the training pipeline first.")

# Tab 3: About
with tab3:
    st.header("About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    This application predicts Drug-Target Interactions (DTI) using machine learning models trained on BindingDB data.
    
    ### 🧬 Features
    **Drug Features (~2927 dimensions):**
    - Morgan ECFP4 Fingerprints (1024 bits)
    - Morgan ECFP6 Fingerprints (1024 bits)
    - Atom Pair Fingerprints (512 bits)
    - MACCS Keys (167 bits)
    - RDKit Descriptors (~200 features)
    
    **Protein Features (567 dimensions):**
    - Amino Acid Composition (20 features)
    - Dipeptide Composition (400 features)
    - CTD Composition (21 features)
    - CTD Transition (21 features)
    - CTD Distribution (105 features)
    
    ### 🤖 Models
    - Logistic Regression
    - Random Forest
    - XGBoost
    - LightGBM
    - Neural Network (MLP)
    - Extra Trees
    - Histogram Gradient Boosting
    - Voting Ensemble
    - Stacking Ensemble
    - CatBoost (if available)
    
    ### 📊 Evaluation Metrics
    - **Accuracy**: Overall classification accuracy
    - **Precision**: True positives / (True positives + False positives)
    - **Recall**: True positives / (True positives + False negatives)
    - **F1-Score**: Harmonic mean of precision and recall
    - **ROC-AUC**: Area under the ROC curve
    
    ### 🔬 Dataset
    Trained on filtered BindingDB data with binary interaction labels (≤1000 nM → interaction).
    
    ### 💻 Technical Details
    - Python 3.10
    - RDKit for molecular descriptors
    - Scikit-learn, XGBoost, LightGBM for ML models
    - Streamlit for web interface
    """)
    
    st.info("💡 **Tip**: For best results, use validated SMILES strings and complete protein sequences.")
    
    st.warning("""
    **⚠️ Model Behavior Note:**
    
    Your models may predict "No Interaction" for most inputs due to:
    - Training data characteristics (class imbalance)
    - Conservative decision threshold (0.5)
    - Protein sequence differences from training set
    
    **Focus on the confidence scores** rather than binary predictions:
    - Interaction confidence >40%: Worth investigating
    - Interaction confidence 25-40%: Uncertain, needs validation
    - Interaction confidence <25%: Likely no interaction
    """)

st.sidebar.title("Navigation")
st.sidebar.info("""
**Quick Start:**
1. Go to 'Predict' tab
2. Enter drug SMILES
3. Enter protein sequence
4. Select a model
5. Click 'Predict Interaction'
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Final Year DTI Prediction")
st.sidebar.markdown("**Framework:** Streamlit + Scikit-learn")
