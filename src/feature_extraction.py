# """
# Feature extraction module for drugs and proteins
# """

# import numpy as np
# import pandas as pd
# import logging
# from rdkit import Chem
# from rdkit.Chem import rdMolDescriptors
# from collections import Counter

# class FeatureExtractor:
#     """Extracts molecular features from drugs and proteins"""
    
#     def __init__(self, config):
#         self.config = config
#         self.logger = logging.getLogger(__name__)
        
#         # Standard amino acids
#         self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
#                            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
#     def extract_features(self, df):
#         """
#         Extract features from the dataset
        
#         Args:
#             df: DataFrame with smiles, protein_sequence, interaction columns
            
#         Returns:
#             tuple: (X, y) feature matrix and target vector
#         """
#         self.logger.info("Extracting molecular features...")
        
#         # Extract drug features
#         drug_features = self.extract_drug_features(df['smiles'])
        
#         # Extract protein features
#         protein_features = self.extract_protein_features(df['protein_sequence'])
        
#         # Combine features
#         X = np.hstack([drug_features, protein_features])
#         y = df['interaction'].values
        
#         self.logger.info(f"Feature matrix shape: {X.shape}")
#         self.logger.info(f"Drug features: {drug_features.shape[1]} dimensions")
#         self.logger.info(f"Protein features: {protein_features.shape[1]} dimensions")
        
#         return X, y
    
#     def extract_drug_features(self, smiles_series):
#         """
#         Extract Morgan fingerprints from SMILES
        
#         Args:
#             smiles_series: pandas Series of SMILES strings
            
#         Returns:
#             np.ndarray: Morgan fingerprint matrix
#         """
#         self.logger.info("Extracting drug Morgan fingerprints...")
        
#         fingerprints = []
#         failed_count = 0
        
#         for smiles in smiles_series:
#             try:
#                 mol = Chem.MolFromSmiles(smiles)
#                 if mol is not None:
#                     # Generate Morgan fingerprint
#                     fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
#                         mol, 
#                         radius=self.config.MORGAN_RADIUS,
#                         nBits=self.config.MORGAN_NBITS
#                     )
#                     # Convert to numpy array
#                     fp_array = np.array(fp)
#                     fingerprints.append(fp_array)
#                 else:
#                     # Invalid SMILES - use zero vector
#                     fingerprints.append(np.zeros(self.config.MORGAN_NBITS))
#                     failed_count += 1
#             except Exception:
#                 # Error processing SMILES - use zero vector
#                 fingerprints.append(np.zeros(self.config.MORGAN_NBITS))
#                 failed_count += 1
        
#         if failed_count > 0:
#             self.logger.warning(f"Failed to process {failed_count} SMILES strings")
        
#         return np.array(fingerprints)
    
#     def extract_protein_features(self, sequence_series):
#         """
#         Extract protein features (amino acid composition + CTD composition)
        
#         Args:
#             sequence_series: pandas Series of protein sequences
            
#         Returns:
#             np.ndarray: Protein feature matrix
#         """
#         self.logger.info("Extracting protein features...")
        
#         features = []
        
#         for sequence in sequence_series:
#             # Amino acid composition (20 features)
#             aa_comp = self.calculate_amino_acid_composition(sequence)
            
#             # CTD composition features (21 features)
#             ctd_comp = self.calculate_ctd_composition(sequence)
            
#             # Combine features
#             protein_features = np.concatenate([aa_comp, ctd_comp])
#             features.append(protein_features)
        
#         return np.array(features)
    
#     def calculate_amino_acid_composition(self, sequence):
#         """
#         Calculate amino acid composition
        
#         Args:
#             sequence: protein sequence string
            
#         Returns:
#             np.ndarray: amino acid composition vector (20 features)
#         """
#         if not sequence or len(sequence) == 0:
#             return np.zeros(20)
        
#         # Count amino acids
#         aa_counts = Counter(sequence.upper())
#         total_length = len(sequence)
        
#         # Calculate composition for each amino acid
#         composition = []
#         for aa in self.amino_acids:
#             composition.append(aa_counts.get(aa, 0) / total_length)
        
#         return np.array(composition)
    
#     def calculate_ctd_composition(self, sequence):
#         """
#         Calculate CTD (Composition, Transition, Distribution) composition features
        
#         Args:
#             sequence: protein sequence string
            
#         Returns:
#             np.ndarray: CTD composition vector (21 features)
#         """
#         if not sequence or len(sequence) == 0:
#             return np.zeros(21)
        
#         # CTD groups based on physicochemical properties
#         ctd_groups = {
#             'hydrophobicity': {
#                 'polar': 'RKEDQN',
#                 'neutral': 'GASTPHY',
#                 'hydrophobic': 'CLVIMFW'
#             },
#             'normalized_vdw': {
#                 'small': 'GASTPDC',
#                 'medium': 'NVEQIL',
#                 'large': 'MHKFRYW'
#             },
#             'polarity': {
#                 'neutral': 'LIFWCMVY',
#                 'polar': 'PATGS',
#                 'charged': 'HQRKNED'
#             },
#             'polarizability': {
#                 'low': 'GASDT',
#                 'medium': 'CPNVEQIL',
#                 'high': 'KMHFRYW'
#             },
#             'charge': {
#                 'positive': 'KR',
#                 'neutral': 'ANCQGHILMFPSTWYV',
#                 'negative': 'DE'
#             },
#             'secondary_structure': {
#                 'helix': 'AEKLQR',
#                 'strand': 'VIYFM',
#                 'coil': 'GSTPCDNH'
#             },
#             'solvent_accessibility': {
#                 'buried': 'ALFCGIVW',
#                 'exposed': 'MSPTHY',
#                 'intermediate': 'NQKRDE'
#             }
#         }
        
#         composition_features = []
        
#         # Calculate composition for each CTD group
#         for group_name, groups in ctd_groups.items():
#             for subgroup_name, amino_acids in groups.items():
#                 count = sum(1 for aa in sequence.upper() if aa in amino_acids)
#                 composition = count / len(sequence)
#                 composition_features.append(composition)
        
#         return np.array(composition_features)


"""
feature_extraction.py  –  Enhanced v2

Feature extraction module for Drug–Target Interaction prediction.

Drug features  (total ~1391):
  - Morgan ECFP4 fingerprints  : 1024 bits
  - MACCS keys                 :  167 bits
  - RDKit molecular descriptors: ~200 physicochemical features

Protein features (total 567):
  - Amino Acid Composition (AAC)   :  20 features
  - Dipeptide Composition (DPC)    : 400 features  ← NEW
  - CTD Composition  (CTD-C)       :  21 features
  - CTD Transition   (CTD-T)       :  21 features  ← NEW
  - CTD Distribution (CTD-D)       : 105 features  ← NEW

Grand total: ~1958 features  (was 1065)
"""

import warnings
import numpy as np
import logging
from collections import Counter

from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, Descriptors
from rdkit.Chem import rdFingerprintGenerator

# Suppress all RDKit C++ log messages (valence errors, parsing warnings, etc.)
RDLogger.DisableLog("rdApp.*")

from src.config import (
    MORGAN_RADIUS,
    MORGAN_NBITS,
    AMINO_ACIDS
)

# ------------------------------------------------------------------
# Module-level constants (computed once at import time)
# ------------------------------------------------------------------

# ECFP4: radius=2, captures local substructures (existing)
_MORGAN_GEN   = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
# ECFP6: radius=3, captures larger substructures — complementary to ECFP4
_ECFP6_GEN    = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
# Atom Pair: encodes distances between atom types — captures 3D-like topology
_ATOMPAIR_GEN = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=512)

# All RDKit descriptor functions — consistent count across all molecules
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _RDKIT_DESC_FUNCS = [(name, fn) for name, fn in Descriptors.descList]

RDKIT_DESC_COUNT = len(_RDKIT_DESC_FUNCS)

# All 400 dipeptide pairs (20 × 20)
_DIPEPTIDES = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]

# CTD property groups — 7 properties, each with 3 subgroups
_CTD_GROUPS = [
    ["RKEDQN",         "GASTPHY",          "CLVIMFW"],   # hydrophobicity
    ["GASTPDC",        "NVEQIL",           "MHKFRYW"],   # Van der Waals volume
    ["LIFWCMVY",       "PATGS",            "HQRKNED"],   # polarity
    ["GASDT",          "CPNVEQIL",         "KMHFRYW"],   # polarizability
    ["KR",             "ANCQGHILMFPSTWYV", "DE"],        # charge
    ["AEKLQR",         "VIYFM",            "GSTPCDNH"],  # secondary structure
    ["ALFCGIVW",       "MSPTHY",           "NQKRDE"],    # solvent accessibility
]


class FeatureExtractor:
    """Extracts numerical features from drugs and proteins."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_features(self, df, training=True):
        """
        Extract combined drug and protein features.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: smiles, protein_sequence, interaction

        Returns
        -------
        X : np.ndarray
            Feature matrix
        y : np.ndarray or None
            Target labels
        """

        self.logger.info("Starting feature extraction (enhanced v2)")

        drug_features    = self._extract_drug_features(df["smiles"])
        protein_features = self._extract_protein_features(df["protein_sequence"])

        # Interaction features: element-wise product of drug[:K] × protein[:K].
        # Using full protein feature size so every protein descriptor
        # is paired with a corresponding drug descriptor.
        K = protein_features.shape[1]           # 567 — use all protein features
        interaction_feat = drug_features[:, :K] * protein_features

        X = np.hstack([drug_features, protein_features, interaction_feat])
        self.logger.info(
            f"Feature matrix shape: {X.shape} "
            f"[drug={drug_features.shape[1]}, protein={protein_features.shape[1]}, "
            f"interaction={interaction_feat.shape[1]}]"
        )

        if training and "interaction" in df.columns:
            y = df["interaction"].values.astype(int)
            return X, y
        else:
            return X, None

    # ------------------------------------------------------------------
    # Drug features
    # ------------------------------------------------------------------

    def _extract_drug_features(self, smiles_series):
        """
        Extract drug features:
          1. Morgan ECFP4  (1024) — local substructures, radius=2
          2. Morgan ECFP6  (1024) — larger substructures, radius=3  ← NEW
          3. Atom Pair     ( 512) — atom distance topology           ← NEW
          4. MACCS keys    ( 167) — structural keys
          5. RDKit descriptors (~200) — physicochemical properties
          Total: ~2927 drug features
        """
        self.logger.info(
            "Extracting drug features (ECFP4 + ECFP6 + AtomPair + MACCS + RDKit)"
        )

        n            = len(smiles_series)
        ecfp4_arr    = np.zeros((n, 1024),            dtype=np.float32)
        ecfp6_arr    = np.zeros((n, 1024),            dtype=np.float32)
        atompair_arr = np.zeros((n, 512),             dtype=np.float32)
        maccs_arr    = np.zeros((n, 167),             dtype=np.float32)
        rdkit_arr    = np.zeros((n, RDKIT_DESC_COUNT),dtype=np.float32)

        failed = 0
        for i, smiles in enumerate(tqdm(smiles_series, desc="Drug features",
                                        unit="mol", ncols=80, leave=True)):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                failed += 1
                continue

            # 1. ECFP4 (radius=2)
            ecfp4_arr[i]    = _MORGAN_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)
            # 2. ECFP6 (radius=3) — captures larger rings, scaffolds
            ecfp6_arr[i]    = _ECFP6_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)
            # 3. Atom Pair — encodes topological distances between atom types
            atompair_arr[i] = _ATOMPAIR_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)
            # 4. MACCS structural keys
            maccs_arr[i]    = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
            # 5. RDKit physicochemical descriptors
            rdkit_arr[i]    = self._rdkit_desc_vector(mol)

        if failed:
            self.logger.warning(f"{failed} invalid SMILES replaced with zero vectors")

        return np.hstack([ecfp4_arr, ecfp6_arr, atompair_arr, maccs_arr, rdkit_arr])

    def _rdkit_desc_vector(self, mol):
        """Compute all RDKit descriptors; replace NaN/Inf/overflow with 0."""
        _F32_MAX = np.finfo(np.float32).max  # ~3.4e38
        values = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _, fn in _RDKIT_DESC_FUNCS:
                try:
                    v = float(fn(mol))
                    # Replace NaN, Inf, or values that overflow float32
                    if not np.isfinite(v) or abs(v) > _F32_MAX:
                        v = 0.0
                except Exception:
                    v = 0.0
                values.append(v)
        return np.array(values, dtype=np.float32)

    # ------------------------------------------------------------------
    # Protein features
    # ------------------------------------------------------------------

    def _extract_protein_features(self, sequence_series):
        """
        Extract protein features:
          - AAC   :  20 features
          - DPC   : 400 features
          - CTD-C :  21 features
          - CTD-T :  21 features
          - CTD-D : 105 features
          Total   : 567 features
        """
        self.logger.info("Extracting protein features (AAC + DPC + CTD-C/T/D)")

        features = []
        for seq in tqdm(sequence_series, desc="Protein features",
                        unit="seq", ncols=80, leave=True):
            aac   = self._amino_acid_composition(seq)
            dpc   = self._dipeptide_composition(seq)
            ctd_c = self._ctd_composition(seq)
            ctd_t = self._ctd_transition(seq)
            ctd_d = self._ctd_distribution(seq)
            features.append(np.concatenate([aac, dpc, ctd_c, ctd_t, ctd_d]))

        return np.array(features, dtype=np.float32)

    def _amino_acid_composition(self, sequence):
        """AAC — 20 features."""
        if not sequence:
            return np.zeros(len(AMINO_ACIDS), dtype=np.float32)
        sequence = sequence.upper()
        total    = len(sequence)
        counts   = Counter(sequence)
        return np.array(
            [counts.get(aa, 0) / total for aa in AMINO_ACIDS],
            dtype=np.float32
        )

    def _dipeptide_composition(self, sequence):
        """
        Dipeptide Composition — 400 features (20 × 20 pairs).
        Captures pairwise amino acid co-occurrence frequencies.
        """
        if not sequence or len(sequence) < 2:
            return np.zeros(400, dtype=np.float32)
        sequence = sequence.upper()
        total    = len(sequence) - 1
        counts   = Counter(sequence[i:i+2] for i in range(total))
        return np.array(
            [counts.get(dp, 0) / total for dp in _DIPEPTIDES],
            dtype=np.float32
        )

    def _ctd_composition(self, sequence):
        """CTD Composition — 21 features (7 properties × 3 subgroups)."""
        if not sequence:
            return np.zeros(21, dtype=np.float32)
        sequence = sequence.upper()
        length   = len(sequence)
        features = []
        for property_groups in _CTD_GROUPS:
            for subset in property_groups:
                count = sum(sequence.count(aa) for aa in subset)
                features.append(count / length)
        return np.array(features, dtype=np.float32)

    def _ctd_transition(self, sequence):
        """
        CTD Transition — 21 features (7 properties × 3 group-pair transitions).

        For each physicochemical property, counts how often adjacent residues
        belong to different groups (G0↔G1, G0↔G2, G1↔G2), normalised by
        the total number of consecutive pairs.
        """
        if not sequence or len(sequence) < 2:
            return np.zeros(21, dtype=np.float32)
        sequence    = sequence.upper()
        length      = len(sequence)
        total_trans = length - 1
        features    = []

        for property_groups in _CTD_GROUPS:
            # Map each amino acid to its group index (0, 1, or 2)
            group_map = {}
            for idx, aa_set in enumerate(property_groups):
                for aa in aa_set:
                    group_map[aa] = idx

            encoded = [group_map.get(aa, -1) for aa in sequence]

            # Bidirectional transition counts for the 3 distinct pairs
            trans = {(0, 1): 0, (0, 2): 0, (1, 2): 0}
            for i in range(total_trans):
                g1, g2 = encoded[i], encoded[i + 1]
                if g1 == -1 or g2 == -1 or g1 == g2:
                    continue
                pair = (min(g1, g2), max(g1, g2))
                if pair in trans:
                    trans[pair] += 1

            for pair in [(0, 1), (0, 2), (1, 2)]:
                features.append(trans[pair] / max(total_trans, 1))

        return np.array(features, dtype=np.float32)

    def _ctd_distribution(self, sequence):
        """
        CTD Distribution — 105 features (7 properties × 3 groups × 5 percentiles).

        For each group records the normalised sequence position at which the
        1st, 25th, 50th, 75th, and 100th percentile member of that group occurs.
        """
        if not sequence:
            return np.zeros(105, dtype=np.float32)
        sequence = sequence.upper()
        length   = len(sequence)
        features = []

        for property_groups in _CTD_GROUPS:
            group_map = {}
            for idx, aa_set in enumerate(property_groups):
                for aa in aa_set:
                    group_map[aa] = idx

            # Collect 0-based positions for each subgroup
            group_positions = [[], [], []]
            for pos, aa in enumerate(sequence):
                g = group_map.get(aa, -1)
                if g >= 0:
                    group_positions[g].append(pos)

            for positions in group_positions:
                if not positions:
                    features.extend([0.0] * 5)
                else:
                    n = len(positions)
                    for pct in [0, 25, 50, 75, 100]:
                        idx = int(np.ceil(pct / 100.0 * n)) - 1
                        idx = max(0, min(idx, n - 1))
                        features.append((positions[idx] + 1) / length)

        return np.array(features, dtype=np.float32)
