"""
Main script for training churn prediction models.

Usage:
    python -m src.train --models CatBoost LGBM --cv-folds 5 --use-smote
    python -m src.train --all-models --experiment-name "Production_Training"
"""
import argparse
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_train_data, split_features_target
from src.training import (
    train_multiple_models, 
    setup_mlflow, 
    get_model_configs
)
from src.evaluation import plot_roc_curves, plot_model_comparison
from src.config import RANDOM_SEED
from src.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train bank churn prediction models"
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['CatBoost', 'LGBM', 'XGBoost', 'Random_Forest', 'MLP_Network', 'Gaussian_NB', 'KNN'],
        help='Models to train (space separated)'
    )
    
    parser.add_argument(
        '--all-models',
        action='store_true',
        help='Train all available models'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5)'
    )
    
    parser.add_argument(
        '--use-smote',
        action='store_true',
        default=True,
        help='Use SMOTE for class balancing (default: True)'
    )
    
    parser.add_argument(
        '--no-smote',
        action='store_true',
        help='Do not use SMOTE'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for validation (default: 0.2)'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='Bank_Churn_Prediction',
        help='Experiment name in MLflow'
    )
    
    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Do not log to MLflow'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save result plots'
    )
    
    parser.add_argument(
        '--no-threshold-wrapper',
        action='store_true',
        help='Do not use threshold wrapper (use default 0.5 threshold)'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("="*70)
    logger.info("MODEL TRAINING START")
    logger.info("="*70)
    
    logger.info("\n[1/5] Loading data...")
    df_train = load_train_data()
    X, y = split_features_target(df_train)
    
    logger.info(f"\n[2/5] Splitting data (test_size={args.test_size})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=args.test_size, 
        random_state=RANDOM_SEED, 
        stratify=y
    )
    logger.info(f"  Train: {X_train.shape[0]} samples")
    logger.info(f"  Validation: {X_val.shape[0]} samples")
    
    if not args.no_mlflow:
        logger.info(f"\n[3/5] Configuring MLflow...")
        setup_mlflow(experiment_name=args.experiment_name)
    else:
        logger.info("\n[3/5] MLflow disabled")
    
    logger.info("\n[4/5] Preparing models...")
    all_models = get_model_configs()
    
    if args.all_models:
        models_to_train = all_models
        logger.info(f"  Training ALL models ({len(models_to_train)})")
    elif args.models:
        models_to_train = {k: v for k, v in all_models.items() if k in args.models}
        logger.info(f"  Training selected models: {list(models_to_train.keys())}")
    else:
        default_models = ['CatBoost', 'LGBM', 'XGBoost']
        models_to_train = {k: v for k, v in all_models.items() if k in default_models}
        logger.info(f"  Training default models: {list(models_to_train.keys())}")
    
    logger.info("\n[5/5] Starting training...")
    use_smote = not args.no_smote if args.no_smote else args.use_smote
    use_threshold_wrapper = not args.no_threshold_wrapper
    
    if not use_threshold_wrapper:
        logger.info("  ⚠ Training WITHOUT threshold wrapper (using default 0.5)")
    
    results_df, roc_data = train_multiple_models(
        models_dict=models_to_train,
        X_train=X_train,
        y_train=y_train,
        n_splits=args.cv_folds,
        use_smote=use_smote,
        log_to_mlflow=not args.no_mlflow,
        use_threshold_wrapper=use_threshold_wrapper
    )
    
    logger.info("\n[6/6] Generating visualizations...")
    
    save_path_roc = None
    save_path_comparison = None
    
    if args.save_plots:
        plots_dir = project_root / 'docs' / 'images'
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path_roc = plots_dir / 'training_roc_curves.png'
        save_path_comparison = plots_dir / 'training_model_comparison.png'
    
    plot_roc_curves(roc_data, save_path=save_path_roc)
    plot_model_comparison(results_df, save_path=save_path_comparison)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETED")
    logger.info("="*70)
    logger.info(f"\nBest model: {results_df.iloc[0]['Model']}")
    logger.info(f"F1-Score: {results_df.iloc[0]['F1_Optimized']:.4f}")
    logger.info(f"Optimal threshold: {results_df.iloc[0]['Threshold']:.4f}")
    logger.info(f"AUC: {results_df.iloc[0]['AUC']:.4f}")
    
    if not args.no_mlflow:
        logger.info(f"\nCheck results in MLflow UI:")
        logger.info(f"  mlflow ui")
    
    logger.info("\n" + "="*70)


if __name__ == "__main__":
    main()
