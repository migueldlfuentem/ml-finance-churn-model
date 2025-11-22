# Guía de Uso - ML Finance Churn Model

Esta guía explica cómo usar los scripts Python para entrenar modelos y generar predicciones.

## 📋 Tabla de Contenidos

- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Entrenamiento de Modelos](#entrenamiento-de-modelos)
- [Generación de Predicciones](#generación-de-predicciones)
- [MLflow UI](#mlflow-ui)
- [Ejemplos Avanzados](#ejemplos-avanzados)

---

## 🔧 Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
python -c "import src; print('✓ Instalación correcta')"
```

---

## Estructura del Proyecto

```
src/
├── data/              # Módulo de carga de datos
│   ├── data_loader.py
│   └── __init__.py
├── features/          # Feature engineering
│   ├── engineering.py
│   └── __init__.py
├── models/            # Pipelines, entrenamiento y evaluación
│   ├── pipelines.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── __init__.py
├── utils/             # Utilidades (MLflow, etc.)
│   ├── mlflow_utils.py
│   └── __init__.py
├── config.py          # Configuración global
├── logger.py          # Sistema de logging
├── train.py           # Script principal de entrenamiento
└── predict.py         # Script principal de predicción
```

---

## Entrenamiento de Modelos

### Uso Básico

```bash
# Entrenar modelos por defecto (CatBoost, LGBM, XGBoost)
python -m src.train
```

### Entrenar Modelos Específicos

```bash
# Entrenar solo CatBoost y LGBM
python -m src.train --models CatBoost LGBM

# Entrenar todos los modelos disponibles
python -m src.train --all-models
```

### Opciones de Configuración

```bash
# Cambiar número de folds de validación cruzada
python -m src.train --cv-folds 10

# Desactivar SMOTE
python -m src.train --no-smote

# Cambiar proporción de validación
python -m src.train --test-size 0.3

# Cambiar nombre del experimento en MLflow
python -m src.train --experiment-name "Production_Training_v2"

# Guardar gráficos de resultados
python -m src.train --save-plots
```

### Entrenar sin MLflow

```bash
python -m src.train --no-mlflow
```

### Ayuda Completa

```bash
python -m src.train --help
```

---

## Generación de Predicciones

### Uso Básico

```bash
# Usar última versión del modelo
python -m src.predict --model-name churn-catboost
```

### Especificar Versión del Modelo

```bash
# Usar versión específica
python -m src.predict --model-name churn-catboost --version 1

# Usar modelo en stage Production
python -m src.predict --model-name churn-catboost --stage Production
```

### Opciones de Salida

```bash
# Especificar nombre del archivo de salida
python -m src.predict --model-name churn-catboost --output my_submission.csv

# Especificar directorio de salida
python -m src.predict --model-name churn-catboost --output-dir ./custom_submissions
```

### Usar Datos de Test Personalizados

```bash
python -m src.predict --model-name churn-catboost --test-data ./data/custom_test.csv
```

### Ayuda Completa

```bash
python -m src.predict --help
```

---

## MLflow UI

### Iniciar MLflow UI

```bash
# Desde el directorio raíz del proyecto
mlflow ui
```

Luego abre tu navegador en: `http://localhost:5000`

### Funcionalidades de MLflow UI

- **Comparar experimentos**: Ver métricas de todos los modelos entrenados
- **Visualizar parámetros**: Revisar hiperparámetros usados
- **Descargar modelos**: Exportar modelos entrenados
- **Model Registry**: Gestionar versiones y stages de modelos

---

## Ejemplos Avanzados

### Pipeline Completo de Entrenamiento y Predicción

```bash
# 1. Entrenar todos los modelos con 10-fold CV y guardar gráficos
python -m src.train --all-models --cv-folds 10 --save-plots

# 2. Ver resultados en MLflow UI
mlflow ui

# 3. Generar predicciones con el mejor modelo
python -m src.predict --model-name churn-catboost --version 1
```

### Experimentación Rápida

```bash
# Entrenar rápido sin MLflow (para debugging)
python -m src.train --models CatBoost --cv-folds 3 --no-mlflow
```

### Producción

```bash
# Entrenar con configuración de producción
python -m src.train \
    --models CatBoost LGBM \
    --cv-folds 10 \
    --experiment-name "Production_v1" \
    --save-plots

# Generar predicciones con modelo en producción
python -m src.predict \
    --model-name churn-catboost \
    --stage Production \
    --output production_submission.csv
```

---

## Uso Programático (Python)

También puedes usar los módulos directamente en Python:

```python
from src.data import load_train_data, split_features_target
from src.training import get_model_configs, train_with_cv, setup_mlflow

# Configurar MLflow
setup_mlflow(experiment_name="My_Experiment")

# Cargar datos
df = load_train_data()
X, y = split_features_target(df)

# Obtener modelos
models = get_model_configs()

# Entrenar un modelo
results = train_with_cv(
    model=models['CatBoost'],
    X_train=X,
    y_train=y,
    model_name='CatBoost',
    n_splits=5
)

print(f"F1-Score: {results['f1_mean']:.4f}")
```

---

## Notas Importantes

1. **Features Dinámicas**: El sistema usa automáticamente las features seleccionadas por Boruta desde `config/selected_features.json`

2. **SMOTE por Defecto**: El balanceo de clases con SMOTE está activado por defecto. Usa `--no-smote` para desactivarlo.

3. **MLflow Database**: Los experimentos se guardan en `mlflow.db` en el directorio raíz.

4. **Logs**: Los logs se guardan en el directorio `logs/`.

5. **Modelos Disponibles**:
   - CatBoost
   - LGBM (LightGBM)
   - XGBoost
   - Random_Forest
   - MLP_Network
   - Gaussian_NB
   - KNN

---

## Troubleshooting

### Error: "No module named 'src'"

```bash
# Asegúrate de ejecutar desde el directorio raíz
cd /path/to/ml-finance-churn-model
python -m src.train
```

### Error al cargar modelo desde MLflow

```bash
# Verifica que el modelo existe
mlflow ui
# Revisa el Model Registry en la UI
```

### Error de memoria con SMOTE

```bash
# Desactiva SMOTE o reduce el tamaño de validación
python -m src.train --no-smote --test-size 0.1
```

---

## Contacto

Para más información, revisa la documentación en `docs/` o los notebooks en `notebooks/`.
