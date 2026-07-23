"""
Utility functions for the DTI Prediction application
"""

import logging
import os
from pathlib import Path

def setup_logging(log_level=logging.INFO):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dti_prediction.log')
        ]
    )

def create_directories():
    """Create necessary directories if they don't exist"""
    from src.config import Config
    
    config = Config()
    
    directories = [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODELS_DIR,
        config.RESULTS_DIR,
        config.PLOTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def validate_environment():
    """
    Validate that all required packages are available
    
    Returns:
        bool: True if environment is valid
    """
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 
        'rdkit', 'xgboost', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {missing_packages}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def print_system_info():
    """Print system information for debugging"""
    import sys
    import platform
    
    print("System Information:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")

def format_time(seconds):
    """
    Format time in seconds to human readable format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"