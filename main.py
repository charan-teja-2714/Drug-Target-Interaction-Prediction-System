# #!/usr/bin/env python3
# """
# Main entry point for DTI Prediction Application
# Orchestrates the complete machine learning pipeline
# """

# import sys
# import os
# import logging
# from pathlib import Path

# # Add src to path for imports
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# from src.config import RANDOM_STATE
# from src.data_loader import DataLoader
# from src.feature_extraction import FeatureExtractor

# from src.data_loader import DataLoader
# from src.feature_extraction import FeatureExtractor
# from src.train import ModelTrainer
# from src.evaluate import ModelEvaluator
# from src.utils import setup_logging, create_directories

# def main():
#     """Main pipeline execution"""
#     # Setup logging
#     setup_logging()
#     logger = logging.getLogger(__name__)
    
#     logger.info("Starting DTI Prediction Pipeline")
    
#     try:
#         # Create necessary directories
#         create_directories()
        
#         # Load configuration
#         # config = Config()
        
#         # Load data
#         logger.info("Loading dataset...")
#         data_loader = DataLoader(config)
#         df = data_loader.load_data()
        
#         # Extract features
#         logger.info("Extracting features...")
#         feature_extractor = FeatureExtractor(config)
#         X, y = feature_extractor.extract_features(df)
        
#         # Train models
#         logger.info("Training models...")
#         trainer = ModelTrainer(config)
#         trained_models = trainer.train_all_models(X, y)
        
#         # Evaluate models
#         logger.info("Evaluating models...")
#         evaluator = ModelEvaluator(config)
#         results = evaluator.evaluate_all_models(trained_models, X, y)
        
#         # Save results
#         evaluator.save_results(results)
        
#         logger.info("Pipeline completed successfully!")
#         print("\nResults saved to results/metrics.csv")
#         print("Trained models saved to models/saved_models/")
        
#     except Exception as e:
#         logger.error(f"Pipeline failed: {str(e)}")
#         print(f"Error: {str(e)}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Main entry point for DTI Prediction Application

This script orchestrates the complete machine learning pipeline:
1. Load dataset
2. Extract features
3. Train ML models
4. Evaluate models
5. Save results

Compatible with:
- Windows
- Python 3.10
"""

import logging

from src.data_loader import DataLoader
from src.feature_extraction import FeatureExtractor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.utils import setup_logging


def main():
    """Main pipeline execution"""

    # --------------------------------------------------
    # Setup logging
    # --------------------------------------------------
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting DTI Prediction Pipeline")

    try:
        # --------------------------------------------------
        # Load dataset
        # --------------------------------------------------
        logger.info("Loading dataset...")
        data_loader = DataLoader()
        df = data_loader.load_data()

        # --------------------------------------------------
        # Feature extraction
        # --------------------------------------------------
        logger.info("Extracting features...")
        feature_extractor = FeatureExtractor()
        X, y = feature_extractor.extract_features(df)

        # --------------------------------------------------
        # Train models
        # --------------------------------------------------
        logger.info("Training models...")
        trainer = ModelTrainer()
        trained_models = trainer.train_all_models(X, y)

        # --------------------------------------------------
        # Evaluate models
        # --------------------------------------------------
        logger.info("Evaluating models...")
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_all_models(trained_models, X, y)

        evaluator.save_results(results)

        logger.info("Pipeline completed successfully!")
        print("\n✔ Results saved to: results/metrics.csv")
        print("✔ Trained models saved to: models/saved_models/")

    except Exception as e:
        logger.exception("Pipeline failed due to an error")
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
