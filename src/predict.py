"""
Script principal para generar predicciones con modelos entrenados.

Uso:
    python -m src.predict --model-name churn-catboost --version 1
    python -m src.predict --model-name churn-catboost --stage Production --output custom_submission.csv
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_test_data
from src.utils import load_model_from_registry
from src.config import SUBMISSION_DIR
from src.logger import get_logger
import mlflow

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generar predicciones de churn con modelo entrenado"
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Nombre del modelo en MLflow Registry (ej: churn-catboost)'
    )
    
    parser.add_argument(
        '--version',
        type=str,
        help='Versión específica del modelo (ej: 1, 2, 3)'
    )
    
    parser.add_argument(
        '--stage',
        type=str,
        choices=['Staging', 'Production', 'Archived'],
        help='Stage del modelo en MLflow'
    )
    
    parser.add_argument(
        '--test-data',
        type=str,
        help='Ruta al archivo CSV de test (opcional, usa config por defecto)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Nombre del archivo de salida (opcional, genera automáticamente)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directorio de salida (opcional, usa submissions/ por defecto)'
    )
    
    return parser.parse_args()


def main():
    """Función principal de predicción."""
    args = parse_args()
    
    logger.info("="*70)
    logger.info("GENERACIÓN DE PREDICCIONES")
    logger.info("="*70)
    
    db_path = project_root / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"\nMLflow tracking URI: {tracking_uri}")
    
    logger.info(f"\n[1/4] Cargando modelo desde MLflow Registry...")
    logger.info(f"  Modelo: {args.model_name}")
    
    if args.version:
        logger.info(f"  Versión: {args.version}")
        model = load_model_from_registry(args.model_name, version=args.version)
    elif args.stage:
        logger.info(f"  Stage: {args.stage}")
        model = load_model_from_registry(args.model_name, stage=args.stage)
    else:
        logger.info(f"  Versión: latest")
        model = load_model_from_registry(args.model_name)
    
    logger.info(f"\n[2/4] Cargando datos de test...")
    if args.test_data:
        test_path = Path(args.test_data)
        df_test = pd.read_csv(test_path)
        logger.info(f"  Datos cargados desde: {test_path}")
    else:
        df_test = load_test_data()
    
    logger.info(f"  Total de muestras: {len(df_test)}")
    
    logger.info(f"\n[3/4] Generando predicciones...")
    
    if 'CustomerId' in df_test.columns:
        customer_ids = df_test['CustomerId']
    else:
        logger.warning("  ⚠ No se encontró columna 'CustomerId', usando índices")
        customer_ids = df_test.index
    
    predictions = model.predict(df_test)
    logger.info(f"  ✓ Predicciones generadas: {len(predictions)}")
    logger.info(f"  Distribución: {pd.Series(predictions).value_counts().to_dict()}")
    
    logger.info(f"\n[4/4] Creando archivo de submission...")
    
    submission = pd.DataFrame({
        'CustomerId': customer_ids,
        'Exited': predictions
    })
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = SUBMISSION_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output:
        output_filename = args.output
    else:
        model_short_name = args.model_name.replace('churn-', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"submission_{model_short_name}_{timestamp}.csv"
    
    output_path = output_dir / output_filename
    submission.to_csv(output_path, index=False)
    
    logger.info(f"  ✓ Archivo guardado en: {output_path}")
    logger.info(f"\nPrimeras filas del submission:")
    print(submission.head(10).to_string(index=False))
    
    logger.info("\n" + "="*70)
    logger.info("PREDICCIONES COMPLETADAS")
    logger.info("="*70)


if __name__ == "__main__":
    main()
