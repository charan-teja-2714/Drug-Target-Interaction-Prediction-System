# """
# Data loading and validation module
# """

# import pandas as pd
# import logging
# from pathlib import Path

# class DataLoader:
#     """Handles data loading and basic validation"""
    
#     def __init__(self, config):
#         self.config = config
#         self.logger = logging.getLogger(__name__)
    
#     def load_data(self):
#         """
#         Load and validate the BindingDB filtered dataset
        
#         Returns:
#             pd.DataFrame: Loaded and validated dataset
#         """
#         try:
#             # Check if file exists
#             if not self.config.INPUT_FILE.exists():
#                 raise FileNotFoundError(f"Dataset not found: {self.config.INPUT_FILE}")
            
#             # Load CSV
#             self.logger.info(f"Loading data from {self.config.INPUT_FILE}")
#             df = pd.read_csv(self.config.INPUT_FILE)
            
#             # Validate required columns
#             missing_cols = set(self.config.REQUIRED_COLUMNS) - set(df.columns)
#             if missing_cols:
#                 raise ValueError(f"Missing required columns: {missing_cols}")
            
#             # Basic data validation
#             initial_rows = len(df)
            
#             # Remove rows with missing values in required columns
#             df = df.dropna(subset=self.config.REQUIRED_COLUMNS)
            
#             # Validate interaction column (should be 0 or 1)
#             if not df['interaction'].isin([0, 1]).all():
#                 raise ValueError("Interaction column must contain only 0 and 1 values")
            
#             # Log data statistics
#             final_rows = len(df)
#             self.logger.info(f"Loaded {final_rows} rows ({initial_rows - final_rows} rows removed due to missing values)")
#             self.logger.info(f"Positive interactions: {df['interaction'].sum()}")
#             self.logger.info(f"Negative interactions: {len(df) - df['interaction'].sum()}")
            
#             return df
            
#         except Exception as e:
#             self.logger.error(f"Error loading data: {str(e)}")
#             raise
    
#     def validate_smiles(self, smiles_series):
#         """
#         Basic SMILES validation
        
#         Args:
#             smiles_series: pandas Series of SMILES strings
            
#         Returns:
#             pd.Series: Boolean mask of valid SMILES
#         """
#         # Basic validation - non-empty strings
#         return smiles_series.notna() & (smiles_series.str.len() > 0)
    
#     def validate_protein_sequence(self, seq_series):
#         """
#         Basic protein sequence validation
        
#         Args:
#             seq_series: pandas Series of protein sequences
            
#         Returns:
#             pd.Series: Boolean mask of valid sequences
#         """
#         # Basic validation - non-empty strings with valid amino acids
#         valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
#         return seq_series.notna() & (seq_series.str.len() > 0) & \
#                seq_series.apply(lambda x: set(str(x).upper()).issubset(valid_aa) if pd.notna(x) else False)

"""
data_loader.py

Data loading and validation module for the DTI Prediction project.
Handles dataset loading, schema validation, cleaning, and sanity checks.

Compatible with:
- Windows
- Python 3.10
"""

import pandas as pd
import logging
from typing import Tuple
from src.config import RAW_DATA_FILE, REQUIRED_COLUMNS


class DataLoader:
    """Handles data loading and strict validation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> pd.DataFrame:
        """
        Load and validate the BindingDB filtered dataset.

        Returns
        -------
        pd.DataFrame
            Cleaned and validated dataset.

        Raises
        ------
        FileNotFoundError
            If dataset file is missing.
        ValueError
            If dataset schema or values are invalid.
        """

        # --------------------------------------------------------------
        # Check file existence
        # --------------------------------------------------------------
        if not RAW_DATA_FILE.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {RAW_DATA_FILE}"
            )

        self.logger.info(f"Loading dataset from {RAW_DATA_FILE}")
        df = pd.read_csv(RAW_DATA_FILE)

        # --------------------------------------------------------------
        # Validate required columns
        # --------------------------------------------------------------
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df = df[REQUIRED_COLUMNS].copy()

        # --------------------------------------------------------------
        # Drop missing values
        # --------------------------------------------------------------
        initial_rows = len(df)
        df.dropna(subset=REQUIRED_COLUMNS, inplace=True)

        # --------------------------------------------------------------
        # Clean SMILES
        # --------------------------------------------------------------
        df["smiles"] = df["smiles"].astype(str).str.strip()
        df = df[df["smiles"] != ""]

        # --------------------------------------------------------------
        # Clean protein sequences
        # --------------------------------------------------------------
        df["protein_sequence"] = df["protein_sequence"].astype(str).str.strip()
        df = df[df["protein_sequence"] != ""]

        # --------------------------------------------------------------
        # Validate interaction labels
        # --------------------------------------------------------------
        if not set(df["interaction"].unique()).issubset({0, 1}):
            raise ValueError("Interaction column must contain only 0 and 1.")

        df["interaction"] = df["interaction"].astype(int)

        # --------------------------------------------------------------
        # Remove duplicate drug–protein pairs
        # --------------------------------------------------------------
        df.drop_duplicates(
            subset=["smiles", "protein_sequence"],
            inplace=True
        )

        df.reset_index(drop=True, inplace=True)

        # --------------------------------------------------------------
        # Final sanity checks
        # --------------------------------------------------------------
        final_rows = len(df)
        if final_rows == 0:
            raise ValueError("Dataset is empty after cleaning.")

        self.logger.info(
            f"Loaded {final_rows} samples "
            f"({initial_rows - final_rows} rows removed)"
        )

        return df

    @staticmethod
    def get_class_distribution(df: pd.DataFrame) -> Tuple[int, int]:
        """
        Get class distribution of the dataset.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        Tuple[int, int]
            (num_negative, num_positive)
        """

        negatives = (df["interaction"] == 0).sum()
        positives = (df["interaction"] == 1).sum()

        logging.info(
            f"Class distribution -> Negative: {negatives}, Positive: {positives}"
        )

        return negatives, positives
