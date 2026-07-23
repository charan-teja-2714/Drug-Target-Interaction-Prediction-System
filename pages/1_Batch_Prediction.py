"""
Batch Prediction Page for DTI Prediction
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.feature_extraction import FeatureExtractor
from src.config import SAVED_MODELS_DIR

st.set_page_config(page_title="Batch Prediction - DTI", page_icon="📋", layout="wide")


@st.cache_resource
def load_models_and_selector():
    """
    Load the variance selector and all classifier models that are compatible
    with the current feature pipeline (same n_features_in_ as selector output).

    Returns
    -------
    models   : dict  name -> fitted sklearn model
    selector : VarianceThreshold | None
    skipped  : list of (name, n_feat_model, n_feat_expected)
    """
    model_dir = Path(SAVED_MODELS_DIR)
    selector = None
    models = {}
    skipped = []

    # Load the variance selector first so we know the expected feature count.
    selector_path = model_dir / "variance_selector.joblib"
    if selector_path.exists():
        selector = joblib.load(selector_path)

    # Number of features the current pipeline produces after selection.
    expected_n = int(selector.get_support().sum()) if selector is not None else None

    # These are never classifiers for the standard pipeline:
    #   variance_selector -> VarianceThreshold (no predict)
    #   lgbm_on_embeddings -> trained on ChemBERTa/ESM2 embeddings (1088 features)
    EXCLUDED = {"variance_selector", "lgbm_on_embeddings"}

    if model_dir.exists():
        for model_file in sorted(model_dir.glob("*.joblib")):
            name = model_file.stem
            if name in EXCLUDED:
                continue

            obj = joblib.load(model_file)

            # Must be a proper classifier with probability support.
            if not (hasattr(obj, "predict") and hasattr(obj, "predict_proba")):
                continue

            # Check feature count compatibility.
            # CatBoost may report n_features_in_ = 0, so we only reject when
            # the value is explicitly positive and wrong.
            n_feat = getattr(obj, "n_features_in_", None)
            if n_feat is not None and n_feat > 0 and expected_n is not None:
                if n_feat != expected_n:
                    skipped.append((name, n_feat, expected_n))
                    continue

            models[name] = obj

    return models, selector, skipped


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.title("📋 Batch Prediction")
st.markdown("Upload a CSV file with multiple drug-target pairs for batch prediction")

st.info("""
**CSV Format Required:**
- Column 1: `smiles` - Drug SMILES notation
- Column 2: `protein_sequence` - Protein FASTA sequence
- Optional: `id` column for tracking
""")

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

models, selector, skipped_models = load_models_and_selector()

# Show which (incompatible) models were silently excluded.
if skipped_models:
    with st.expander(f"ℹ️ {len(skipped_models)} model(s) excluded (feature mismatch)"):
        st.markdown(
            "These models were trained on a different feature set and cannot be "
            "used with the current pipeline. Re-run `python main.py` to retrain them."
        )
        for name, n_feat, exp_n in skipped_models:
            st.write(f"- **{name}**: trained on {n_feat} features, pipeline now produces {exp_n}")

if models:
    # Only show the top-5 models ranked by ROC-AUC from the evaluation results.
    # (stacking 0.888, voting 0.885, lightgbm 0.884, xgboost 0.881, rf 0.876)
    TOP_MODELS = [
        "lightgbm_tuned",
        "xgboost_tuned",
        "voting_tuned",
        "stacking_tuned",
        "random_forest_tuned",
    ]
    display_models = [m for m in TOP_MODELS if m in models]

    # Clean display labels (strip _tuned suffix)
    display_labels = {m: m.replace("_tuned", "").replace("_", " ").title() for m in display_models}
    label_to_key   = {v: k for k, v in display_labels.items()}

    selected_label = st.selectbox(
        "Select Model",
        options=list(display_labels.values()),
        help="Showing top 5 models ranked by ROC-AUC on the test set.",
    )
    model_name = label_to_key[selected_label]

    # Per-model ROC-AUC badge and guidance.
    MODEL_STATS = {
        "lightgbm_tuned":   ("LightGBM",     "88.4%"),
        "xgboost_tuned":    ("XGBoost",       "88.1%"),
        "voting_tuned":     ("Voting Ensemble","88.5%"),
        "stacking_tuned":   ("Stacking",      "88.8%"),
        "random_forest_tuned": ("Random Forest","87.6%"),
    }
    _, auc = MODEL_STATS.get(model_name, ("", ""))

    if "stacking" in model_name:
        st.warning(
            f"**Stacking** — ROC-AUC {auc} (best overall). "
            "Uses passthrough=True so the meta-learner sees all ~3800 features; "
            "may be less reliable on drug-protein pairs very different from training data. "
            "**lightgbm_tuned** or **xgboost_tuned** are safer for novel inputs."
        )
    else:
        st.success(f"**{selected_label}** selected — ROC-AUC {auc} on test set.")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.subheader("Preview Data")
            st.dataframe(df.head(), use_container_width=True)

            required_cols = ['smiles', 'protein_sequence']
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV must contain columns: {required_cols}")
            else:
                st.success(f"✅ Loaded {len(df)} drug-target pairs")

                if st.button("🚀 Run Batch Prediction", type="primary"):
                    with st.spinner("Extracting features and running predictions..."):
                        try:
                            # Feature extraction (interaction column required by extractor
                            # but not used when training=False).
                            df['interaction'] = 0
                            extractor = FeatureExtractor()
                            X, _ = extractor.extract_features(df, training=False)

                            # Apply feature selection.
                            if selector is not None:
                                X = selector.transform(X)

                            # Sanity check: verify dimensions match the model.
                            model = models[model_name]
                            n_feat_model = getattr(model, "n_features_in_", None)
                            if (n_feat_model is not None and n_feat_model > 0
                                    and X.shape[1] != n_feat_model):
                                st.error(
                                    f"Feature mismatch: **{model_name}** expects "
                                    f"{n_feat_model} features but the pipeline produced "
                                    f"{X.shape[1]}. Please retrain with `python main.py`."
                                )
                                st.stop()

                            predictions = model.predict(X)
                            probas      = model.predict_proba(X)

                            # Map probability columns using model.classes_ to handle
                            # any class ordering (most models use [0, 1], but be safe).
                            classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
                            idx_no  = classes.index(0) if 0 in classes else 0
                            idx_yes = classes.index(1) if 1 in classes else 1

                            results_df = df[['smiles', 'protein_sequence']].copy()
                            if 'id' in df.columns:
                                results_df.insert(0, 'id', df['id'])

                            results_df['prediction'] = [
                                'Interaction' if p == 1 else 'No Interaction'
                                for p in predictions
                            ]
                            results_df['confidence_no_interaction'] = probas[:, idx_no]
                            results_df['confidence_interaction']    = probas[:, idx_yes]

                            st.success("✅ Predictions Complete!")

                            st.subheader("Results")
                            st.dataframe(results_df, use_container_width=True)

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Predictions", len(predictions))
                            with col2:
                                interactions = int((predictions == 1).sum())
                                st.metric("Predicted Interactions", interactions)
                            with col3:
                                no_interactions = int((predictions == 0).sum())
                                st.metric("Predicted No Interactions", no_interactions)

                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv,
                                file_name="dti_predictions.csv",
                                mime="text/csv",
                            )

                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                            st.exception(e)

        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

    else:
        st.markdown("### Sample CSV Format")
        st.markdown(
            "The sample below contains **3 known interactions** (rows 1–3) and "
            "**3 known non-interactions** (rows 4–6) taken from the BindingDB training set. "
            "Use it to verify the model is working correctly — rows 1–3 should be "
            "predicted as **Interaction** and rows 4–6 as **No Interaction**."
        )
        # Real drug-protein pairs from the BindingDB training set.
        # Rows 1-3: known interactions (label = 1)
        # Rows 4-6: known non-interactions (label = 0)
        # All top models (LightGBM, XGBoost, Stacking) agree on all 6 rows.
        sample_data = """id,smiles,protein_sequence
1,CCc1c(N2CCN(c3ccncn3)CC2)c(=O)n2nc(C3=CCOCC3)nc2n1CC(=O)Nc1ccc(C(F)(F)F)cc1Cl,ENLNSGTVEPTHSKCLKMERNLGLPTKEEEEDDENEANEGEEDDDKDFLWPAPNEEQVTCLKMYFGHSSFKPVQWKVIHSVLEERRDNVAVMATGYGKSLCFQYPPVYVGKIGLVISPLISLMEDQVLQLKMSNIPACFLGSAQSENVLTDIKLGKYRIVYVTPEYCSGNMGLLQQLEADIGITLIAVDEAHCISEWGHDFRDSFRKLGSLKTALPMVPIVALTATASSSIREDIVRCLNLRNPQITCTGFDRPNLYLEVRRKTGNILQDLQPFLVKTSSHWEFEGPTIIYCPSRKMTQQVTGELRKLNLSCGTYHAGMSFSTRKDIHHRFVRDEIQCVIATIAFGMGINKADIRQVIHYGAPKDMESYYQEIGRAGRDGLQSSCHVLWAPADINLNRHLLTEIRNEKFRLYKLKMMAKMEKYLHSSRCRRQIILSHFEDKQVQKASLGIMGTEKCCDNCRSRLDHCYSMDDSEDTSWDFGPQAFKLLSAVDILGEKFGIGLPILFLRGSNSQRLADQYRRHSLFGTGKDQTESWWKAFSRQLITEGFLVEVSRYNKFMKICALTKKGRNWLHKANTESQSLILQANEELCPKKLLLPSSKTVSSGTKEHCYNQVPVELSTEKKSNLEKLYSYKPCDKISSGSNISKKSIMVQSPEKAYSSSQPVISAQEQETQIVLYGKLVEARQKHANKMDVPPAILATNKILVDMAKMRPTTVENVKRIDGVSEGKAAMLAPLLEVIKHFCQTNSVQTDLFSSTKPQEEQKTSLVAKNK
2,COCCN1CCN(CC1)C(=O)c1ccc(Nc2nc3c(cccn3n2)N2CCC(O)(CC2)c2ccc(Cl)cc2)cc1,MQYLNIKEDCNAMAFCAKMRSSKKTEVNLEAPEPGVEVIFYLSDREPLRLGSGEYTAEELCIRAAQACRISPLCHNLFALYDENTKLWYAPNRTITVDDKMSLRLHYRMRFYFTNWHGTNDNEQSVWRHSPKKQKNGYEKKKIPDATPLLDASSLEYLFAQGQYDLVKCLAPIRDPKTEQDGHDIENECLGMAVLAISHYAMMKKMQLPELPKDISYKRYIPETLNKSIRQRNLLTRMRINNVFKDFLKEFNNKTICDSSVSTHDLKVKYLATLETLTKHYGAEIFETSMLLISSENEMNWFHSNDGGNVLYYEVMVTGNLGIQWRHKPNVVSVEKEKNKLKRKKLENKHKKDEEKNKIREEWNNFSYFPEITHIVIKESVVSINKQDNKKMELKLSSHEEALSFVSLVDGYFRLTADAHHYLCTDVAPPLIVHNIQNGCHGPICTEYAINKLRQEGSEEGMYVLRWSCTDFDNILMTVTCFEKSEQVQGAQKQFKNFQIEVQKGRYSLHGSDRSFPSLGDLMSHLKKQILRTDNISFMLKRCCQPKPREISNLLVATKKAQEWQPVYPMSQLSFDRILKKDLVQGEHLGRGTRTHIYSGTLMDYKDDEGTSEEKKIKVILKVLDPSHRDISLAFFEAASMMRQVSHKHIVYLYGVCVRDVENIMVEEFVEGGPLDLFMHRKSDVLTTPWKFKVAKQLASALSYLEDKDLVHGNVCTKNLLLAREGIDSECGPFIKLSDPGIPITVLSRQECIERIPWIAPECVEDSKNLSVAADKWSFGTTLWEICYNGEIPLKDKTLIEKERFYESRCRPVTPSCKELADLMTRCMNYDPNQRPFFRAIMRDINKLEEQNPDIVSEKKPATEVDPTHFEKRFLKRIRDLGEGHFGKVELCRYDPEGDNTGEQVAVKSLKPESGGNHIADLKKEIEILRNLYHENIVKYKGICTEDGGNGIKLIMEFLPSGSLKEYLPKNKNKINLKQQLKYAVQICKGMDYLGSRQYVHRDLAARNVLVESEHQVKIGDFGLTKAIETDKEYYTVKDDRDSPVFWYAPECLMQSKFYIASDVWSFGVTLHELLTYCDSDSSPMALFLKMIGPTHGQMTVTRLVNTLKEGKRLPCPPNCPDEVYQLMRKCWEFQPSNRTSFQNLIEGFEALLK
3,COc1cc2[nH]c(=O)c(cc2cc1Cl)[C@H](C)Nc1ccc(C#N)c(C)n1,MSKKISGGSVVEMQGDEMTRIIWELIKEKLIFPYVELDLHSYDLGIENRDATNDQVTKDAAEAIKKHNVGVKCATITPDEKRVEEFKLKQMWKSPNGTIRNILGGTVFREAIICKNIPRLVSGWVKPIIIGHHAYGDQYRATDFVVPGPGKVEITYTPSDGTQKVTYLVHNFEEGGGVAMGMYNQDKSIEDFAHSSFQMALSKGWPLYLSTKNTILKKYDGRFKDIFQEIYDKQYKSQFEAQKIWYEHRLIDDMVAQAMKSEGGFIWACKNYDGDVQSDSVAQGYGSLGMMTSVLVCPDGKTVEAEAAHGTVTRHYRMYQKGQETSTNPIASIFAWTRGLAHRAKLDNNKELAFFANALEEVSIETIEAGFMTKDLAACIKGLPNVQRSDYLNTFEFMDKLGENLKIKLAQAKL
4,Brc1ccc(o1)C(=O)Nc1nc2ccccc2[nH]1,MADSGLDKKSTKCPDCSSASQKDVLCVCSSKTRVPPVLVVEMSQTSSIGSAESLISLERKKEKNINRDITSRKDLPSRTSNVERKASQQQWGRGNFTEGKVPHIRIENGAAIEEIYTFGRILGKGSFGIVIEATDKETETKWAIKKVNKEKAGSSAVKLLEREVNILKSVKHEHIIHLEQVFETPKKMYLVMELCEDGELKEILDRKGHFSENETRWIIQSLASAIAYLHNNDIVHRDLKLENIMVKSSLIDDNNEINLNIKVTDFGLAVKKQSRSEAMLQATCGTPIYMAPEVISAHDYSQQCDIWSIGVVMYMLLRGEPPFLASSEEKLFELIRKGELHFENAVWNSISDCAKSVLKQLMKVDPAHRITAKELLDNQWLTGNKLSSVRPTNVLEMMKEWKNNPESVEENTTEEKNKPSTEEKLKSYQPWGNVPDANYTSDEEEEKQSTAYEKQFPATSKDNFDMCSSSFTSSKLLPAEIKGEMEKTPVTPSQGTATKYPAKSGALSRTKKKL
5,C1Sc2nnc(-c3cc(n[nH]3)-c3ccccc3)n2N=C1c1ccncc1,MSQWYELQQLDSKFLEQVHQLYDDSFPMEIRQYLAQWLEKQDWEHAANDVSFATIRFHDLLSQLDDQYSRFSLENNFLLQHNIRKSKRNLQDNFQEDPIQMSMIIYSCLKEERKILENAQRFNQAQSGNIQSTVMLDKQKELDSKVRNVKDKVMCIEHEIKSLEDLQDEYDFKCKTLQNREHETNGVAKSDQKQEQLLLKKMYLMLDNKRKEVVHKIIELLNVTELTQNALINDELVEWKRRQQSACIGGPPNACLDQLQNWFTIVAESLQQVRQQLKKLEELEQKYTYEHDPITKNKQVLWDRTFSLFQQLIQSSFVVERQPCMPTHPQRPLVLKTGVQFTVKLRLLVKLQELNYNLKVKVLFDKDVNERNTVKGFRKFNILGTHTKVMNMEESTNGSLAAEFRHLQLKEQKNAGTRTNEGPLIVTEELHSLSFETQLCQPGLVIDLETTSLPVVVISNVSQLPSGWASILWYNMLVAEPRNLSFFLTPPCARWAQLSEVLSWQFSSVTKRGLNVDQLNMLGEKLLGPNASPDGLIPWTRFCKENINDKNFPFWLWIESILELIKKHLLPLWNDGCIMGFISKERERALLKDQQPGTFLLRFSESSREGAITFTWVERSQNGGEPDFHAVEPYTKKELSAVTFPDIIRNYKVMAAENIPENPLKYLYPNIDKDHAFGKYYSRPKEAPEPMELDGPKGTGYIKTELISVSEVHPSRLQTTDNLLPMSPEEFDEVSRIVGSVEFDSMMNTV
6,Cc1nnc2ccc3n(Cc4cccc(Cl)c4)c(cc3n12)C1=CC=[N](N1)C1(CC#N)CNC1,STNPPPPETSNPNKPKRQTNQLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTIKKRLENNYYWNAQECIQDFNTMFTNCYIYNKPGDDIVLMAEALEKLFLQKINELPTEE"""

        st.code(sample_data, language='csv')

        st.download_button(
            label="📥 Download Sample CSV",
            data=sample_data,
            file_name="sample_dti_input.csv",
            mime="text/csv",
        )

else:
    st.warning("⚠️ No trained models found. Please run `python main.py` first to train models.")
