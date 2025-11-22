"""
Script de prueba rápida para verificar que los módulos funcionan correctamente.
"""
import sys
from pathlib import Path

# Añadir root al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*70)
print("TEST DE MÓDULOS - ML Finance Churn Model")
print("="*70)

# Test 1: Importar módulos
print("\n[1/5] Testeando imports...")
try:
    from src.data import load_train_data, split_features_target
    from src.features import FeatureEngineer, DropFeatures
    from src.preprocessing import create_full_pipeline
    from src.training import get_model_configs
    from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
    print("  ✓ Todos los imports correctos")
except Exception as e:
    print(f"  ✗ Error en imports: {e}")
    sys.exit(1)

# Test 2: Cargar datos
print("\n[2/5] Testeando carga de datos...")
try:
    df = load_train_data()
    X, y = split_features_target(df)
    print(f"  ✓ Datos cargados: {X.shape[0]} filas, {X.shape[1]} columnas")
    print(f"  ✓ Target: {y.value_counts().to_dict()}")
except Exception as e:
    print(f"  ✗ Error cargando datos: {e}")
    sys.exit(1)

# Test 3: Feature Engineering
print("\n[3/5] Testeando feature engineering...")
try:
    engineer = FeatureEngineer()
    X_engineered = engineer.fit_transform(X.head(100))
    print(f"  ✓ Features originales: {X.shape[1]}")
    print(f"  ✓ Features después de engineering: {X_engineered.shape[1]}")
    print(f"  ✓ Nuevas features creadas: {X_engineered.shape[1] - X.shape[1]}")
except Exception as e:
    print(f"  ✗ Error en feature engineering: {e}")
    sys.exit(1)

# Test 4: Pipeline
print("\n[4/5] Testeando creación de pipeline...")
try:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    pipeline = create_full_pipeline(model, use_smote=False)
    print(f"  ✓ Pipeline creado con {len(pipeline.steps)} pasos:")
    for step_name, _ in pipeline.steps:
        print(f"    - {step_name}")
except Exception as e:
    print(f"  ✗ Error creando pipeline: {e}")
    sys.exit(1)

# Test 5: Entrenamiento rápido
print("\n[5/5] Testeando entrenamiento rápido...")
try:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X.head(500), y.head(500), test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"  ✓ Modelo entrenado")
    print(f"  ✓ Accuracy en test: {score:.4f}")
except Exception as e:
    print(f"  ✗ Error en entrenamiento: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ TODOS LOS TESTS PASARON CORRECTAMENTE")
print("="*70)
print("\nPróximos pasos:")
print("  1. Entrenar modelos: python -m src.train")
print("  2. Ver resultados: mlflow ui")
print("  3. Generar predicciones: python -m src.predict --model-name churn-catboost --version 1")
print("\nRevisa USAGE.md para más información.")
print("="*70)
