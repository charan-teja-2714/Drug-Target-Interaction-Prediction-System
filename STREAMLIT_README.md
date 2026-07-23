# Running the Streamlit Frontend

## Quick Start

1. **Install Streamlit** (if not already installed):
   ```cmd
   pip install streamlit
   ```

2. **Run the application**:
   ```cmd
   streamlit run app.py
   ```

3. **Access the app**:
   - The app will automatically open in your browser at `http://localhost:8501`

## Features

### 🔮 Predict Tab
- Enter drug SMILES and protein sequence
- Select trained model
- Get instant predictions with confidence scores
- Visual probability distribution

### 📋 Batch Prediction Page
- Upload CSV with multiple drug-target pairs
- Process bulk predictions
- Download results as CSV
- Summary statistics

### 📊 Model Performance Tab
- View all model metrics
- Compare model performance
- Interactive charts
- Identify best models

### ℹ️ About Tab
- Feature descriptions
- Model information
- Technical details

## CSV Format for Batch Prediction

```csv
id,smiles,protein_sequence
1,CC(=O)OC1=CC=CC=C1C(=O)O,MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL
2,CN1C=NC2=C1C(=O)N(C(=O)N2C)C,MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL
```

## Notes

- Ensure models are trained before using the app (`python main.py`)
- Models are loaded from `models/saved_models/`
- Metrics are loaded from `results/metrics_tuned_rf.csv`
