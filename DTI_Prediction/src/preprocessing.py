"""
Data preprocessing module
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Handles data preprocessing and splitting"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def preprocess_data(self, df):
        """
        Preprocess the dataset
        
        Args:
            df: pandas DataFrame with raw data
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        self.logger.info("Preprocessing data...")
        
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Clean SMILES strings
        df_processed['smiles'] = df_processed['smiles'].str.strip()
        
        # Clean protein sequences
        df_processed['protein_sequence'] = df_processed['protein_sequence'].str.strip().str.upper()
        
        # Remove any remaining invalid entries
        initial_count = len(df_processed)
        
        # Remove empty SMILES or sequences
        df_processed = df_processed[
            (df_processed['smiles'].str.len() > 0) & 
            (df_processed['protein_sequence'].str.len() > 0)
        ]
        
        final_count = len(df_processed)
        self.logger.info(f"Removed {initial_count - final_count} invalid entries")
        
        return df_processed
    
    def split_data(self, X, y):
        """
        Split data into training and testing sets
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        self.logger.info(f"Splitting data with test_size={self.config.TEST_SIZE}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        self.logger.info(f"Training set: {len(X_train)} samples")
        self.logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test