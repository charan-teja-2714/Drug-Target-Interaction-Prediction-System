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
feature_extraction.py

Feature extraction module for Drug–Target Interaction prediction.
Extracts:
- Drug features using Morgan fingerprints (RDKit)
- Protein features using Amino Acid Composition (AAC) and
  CTD Composition (physicochemical grouping)

Compatible with:
- Windows
- Python 3.10
"""

import numpy as np
import logging
from collections import Counter
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from src.config import (
    MORGAN_RADIUS,
    MORGAN_NBITS,
    AMINO_ACIDS
)


class FeatureExtractor:
    """Extracts numerical features from drugs and proteins."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_features(self, df,training=True):
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
        y : np.ndarray
            Target labels
        """

        self.logger.info("Starting feature extraction")

        drug_features = self._extract_drug_features(df["smiles"])
        protein_features = self._extract_protein_features(df["protein_sequence"])

        X = np.hstack([drug_features, protein_features])
        self.logger.info(f"Final feature matrix shape: {X.shape}")
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
        Extract Morgan fingerprints from SMILES strings.
        Invalid SMILES are converted to zero vectors.
        """

        self.logger.info("Extracting drug fingerprints")

        features = np.zeros((len(smiles_series), MORGAN_NBITS), dtype=np.float32)
        failed = 0

        for i, smiles in enumerate(smiles_series):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failed += 1
                continue

            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol,
                radius=MORGAN_RADIUS,
                nBits=MORGAN_NBITS
            )
            features[i] = np.array(fp)

        if failed > 0:
            self.logger.warning(f"{failed} invalid SMILES replaced with zero vectors")

        return features

    # ------------------------------------------------------------------
    # Protein features
    # ------------------------------------------------------------------
    def _extract_protein_features(self, sequence_series):
        """
        Extract protein features using AAC + CTD composition.
        """

        self.logger.info("Extracting protein features")

        features = []

        for seq in sequence_series:
            aac = self._amino_acid_composition(seq)
            ctd = self._ctd_composition(seq)
            features.append(np.concatenate([aac, ctd]))

        return np.array(features, dtype=np.float32)

    def _amino_acid_composition(self, sequence):
        """
        Amino Acid Composition (20 features).
        """

        if not sequence:
            return np.zeros(len(AMINO_ACIDS), dtype=np.float32)

        sequence = sequence.upper()
        total = len(sequence)
        counts = Counter(sequence)

        return np.array(
            [counts.get(aa, 0) / total for aa in AMINO_ACIDS],
            dtype=np.float32
        )

    def _ctd_composition(self, sequence):
        """
        CTD Composition features (21 features).
        """

        if not sequence:
            return np.zeros(21, dtype=np.float32)

        sequence = sequence.upper()
        length = len(sequence)

        groups = {
            "hydrophobicity": ["RKEDQN", "GASTPHY", "CLVIMFW"],
            "vdw_volume": ["GASTPDC", "NVEQIL", "MHKFRYW"],
            "polarity": ["LIFWCMVY", "PATGS", "HQRKNED"],
            "polarizability": ["GASDT", "CPNVEQIL", "KMHFRYW"],
            "charge": ["KR", "ANCQGHILMFPSTWYV", "DE"],
            "secondary_structure": ["AEKLQR", "VIYFM", "GSTPCDNH"],
            "solvent_accessibility": ["ALFCGIVW", "MSPTHY", "NQKRDE"],
        }

        features = []
        for group in groups.values():
            for subset in group:
                count = sum(sequence.count(aa) for aa in subset)
                features.append(count / length)

        return np.array(features, dtype=np.float32)
