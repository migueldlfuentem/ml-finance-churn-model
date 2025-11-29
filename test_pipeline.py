"""
Quick test script to verify that modules work correctly.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*70)
print("MODULE TEST - ML Finance Churn Model")
print("="*70)

print("\n[1/5] Testing imports...")
try:
    from src.data import load_train_data, split_features_target
    from src.features import FeatureEngineer, DropFeatures
    from src.preprocessing import create_full_pipeline
    from src.training import get_model_configs
    from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

print("\n[2/5] Testing data loading...")
try:
    df = load_train_data()
    X, y = split_features_target(df)
    print(f"  ✓ Data loaded: {X.shape[0]} rows, {X.shape[1]} columns")
    print(f"  ✓ Target: {y.value_counts().to_dict()}")
except Exception as e:
    print(f"  ✗ Data loading error: {e}")
    sys.exit(1)

print("\n[3/5] Testing feature engineering...")
try:
    engineer = FeatureEngineer()
    X_engineered = engineer.fit_transform(X.head(100))
    print(f"  ✓ Original features: {X.shape[1]}")
    print(f"  ✓ Features after engineering: {X_engineered.shape[1]}")
    print(f"  ✓ New features created: {X_engineered.shape[1] - X.shape[1]}")
except Exception as e:
    print(f"  ✗ Feature engineering error: {e}")
    sys.exit(1)

print("\n[4/5] Testing pipeline creation...")
try:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    pipeline = create_full_pipeline(model, use_smote=False)
    print(f"  ✓ Pipeline created with {len(pipeline.steps)} steps:")
    for step_name, _ in pipeline.steps:
        print(f"    - {step_name}")
except Exception as e:
    print(f"  ✗ Pipeline creation error: {e}")
    sys.exit(1)

print("\n[5/5] Testing quick training...")
try:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X.head(500), y.head(500), test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"  ✓ Model trained")
    print(f"  ✓ Test accuracy: {score:.4f}")
except Exception as e:
    print(f"  ✗ Training error: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED SUCCESSFULLY")
print("="*70)
print("\nNext steps:")
print("  1. Train models: python -m src.train")
print("  2. View results: mlflow ui")
print("  3. Generate predictions: python -m src.predict --model-name churn-catboost --version 1")
print("\nCheck USAGE.md for more information.")
print("="*70)
