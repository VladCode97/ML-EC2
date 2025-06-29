# Bankruptcy Prediction API

Sistema modular de predicción de bancarrota empresarial usando machine learning, con API REST construida en FastAPI.

## Estructura del Proyecto

```
MLOpsWorkshop2/
├── main.py              # Pipeline modular de ML
├── api.py               # Aplicación FastAPI
├── Script.py            # Script original de análisis
├── requirements.txt     # Dependencias del proyecto
├── README.md           # Este archivo
├── test_main.http      # Archivo de pruebas HTTP
└── models/             # Directorio donde se guardan los modelos (se crea automáticamente)
```

## Características

### Pipeline Modular (`main.py`)

- **DataPreprocessor**: Clase para preprocesamiento de datos
  - Imputación de valores faltantes con KNN
  - Escalado de características
  - Balanceo de clases con SMOTE
  - Guardado/carga de preprocesadores

- **ModelTrainer**: Clase para entrenamiento de modelos
  - Entrenamiento de múltiples algoritmos (LogisticRegression, KNN, SVM, RandomForest, XGBoost)
  - Optimización de hiperparámetros con GridSearchCV
  - Evaluación automática con métricas de recall
  - Guardado/carga de modelos

- **BankruptcyPredictor**: Clase principal
  - Pipeline completo de entrenamiento
  - Predicciones individuales y en lote
  - Persistencia completa del pipeline

### API REST (`api.py`)

- **Endpoints principales**:
  - `GET /`: Información de la API
  - `GET /health`: Estado de salud del modelo
  - `GET /model/info`: Información del modelo cargado
  - `POST /train`: Entrenar nuevo modelo
  - `POST /predict`: Predicción individual
  - `POST /predict/batch`: Predicciones en lote
  - `GET /features/required`: Features requeridas

- **Validación de datos** con Pydantic
- **Documentación automática** con Swagger UI
- **Manejo de errores** robusto

## Instalación

1. **Clonar el repositorio**:
```bash
git clone <repository-url>
cd MLOpsWorkshop2
```

2. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

3. **Preparar datos**:
   - Asegúrate de que el archivo `Taller2/data.csv` existe
   - O modifica la ruta en el código según tu estructura

## Uso

### 1. Entrenamiento del Modelo

#### Opción A: Usando el script principal
```python
from main import train_bankruptcy_model

# Entrenar y guardar modelo
predictor = train_bankruptcy_model("Taller2/data.csv", "models/")
```

#### Opción B: Usando la API
```bash
# Iniciar la API
python api.py

# Entrenar modelo via API
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{"data_path": "Taller2/data.csv"}'
```

### 2. Predicciones

#### Usando el pipeline directamente:
```python
from main import load_bankruptcy_model

# Cargar modelo entrenado
predictor = load_bankruptcy_model("models/")

# Datos de ejemplo
company_data = {
    "Net Income to Total Assets": 0.1,
    "ROA(A) before interest and % after tax": 0.05,
    "ROA(B) before interest and depreciation after tax": 0.03,
    "ROA(C) before interest and depreciation before interest": 0.02,
    "Net worth/Assets": 0.8
}

# Realizar predicción
result = predictor.predict_single(company_data)
print(f"Predicción: {result['prediction']}")
print(f"Probabilidad: {result['probability']:.3f}")
```

#### Usando la API:
```bash
# Predicción individual
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "net_income_to_total_assets": 0.1,
       "roa_a_before_interest_and_percent_after_tax": 0.05,
       "roa_b_before_interest_and_depreciation_after_tax": 0.03,
       "roa_c_before_interest_and_depreciation_before_interest": 0.02,
       "net_worth_assets": 0.8
     }'

# Predicciones en lote
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "companies": [
         {
           "net_income_to_total_assets": 0.1,
           "roa_a_before_interest_and_percent_after_tax": 0.05,
           "roa_b_before_interest_and_depreciation_after_tax": 0.03,
           "roa_c_before_interest_and_depreciation_before_interest": 0.02,
           "net_worth_assets": 0.8
         }
       ]
     }'
```

### 3. Documentación de la API

Una vez que la API esté corriendo, puedes acceder a:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Estructura de Datos

### Features Requeridas

El modelo requiere las siguientes features principales:

1. `net_income_to_total_assets`: Ingresos netos a activos totales
2. `roa_a_before_interest_and_percent_after_tax`: ROA(A) antes de intereses y después de impuestos
3. `roa_b_before_interest_and_depreciation_after_tax`: ROA(B) antes de intereses y depreciación después de impuestos
4. `roa_c_before_interest_and_depreciation_before_interest`: ROA(C) antes de intereses y depreciación antes de intereses
5. `net_worth_assets`: Valor neto/Activos

### Respuesta de Predicción

```json
{
  "prediction": 0,
  "probability": 0.123,
  "model_used": "LogisticRegression",
  "risk_level": "BAJO"
}
```

Donde:
- `prediction`: 0 (no bancarrota) o 1 (bancarrota)
- `probability`: Probabilidad de bancarrota (0-1)
- `model_used`: Nombre del modelo utilizado
- `risk_level`: "BAJO", "MEDIO", o "ALTO"

## Métricas de Evaluación

El sistema utiliza **Recall** como métrica principal para optimización, ya que:
- Es crítico no perder casos reales de bancarrota
- Se prefiere tener falsas alarmas que perder casos reales
- El dataset está desbalanceado (~3% de casos positivos)

## Configuración

### Variables de Entorno (Opcional)

```bash
export MODEL_PATH="models/"
export DATA_PATH="Taller2/data.csv"
export API_HOST="0.0.0.0"
export API_PORT="8000"
```

### Personalización

Puedes modificar:
- **Hiperparámetros** en `ModelTrainer.get_model_configs()`
- **Métricas de evaluación** en `ModelTrainer.evaluate_models()`
- **Features por defecto** en `map_company_data_to_features()`
- **Umbrales de riesgo** en `get_risk_level()`

## Desarrollo

### Ejecutar Tests

```bash
# Verificar que la API responde
curl http://localhost:8000/health

# Verificar información del modelo
curl http://localhost:8000/model/info
```

### Logs

El sistema incluye logging detallado:
- Entrenamiento de modelos
- Predicciones
- Errores y excepciones

Los logs se muestran en la consola y pueden ser redirigidos a archivos.

## Troubleshooting

### Problemas Comunes

1. **"Model not loaded"**: El modelo no se ha entrenado. Ejecuta `/train` primero.

2. **"Features missing"**: Asegúrate de que el archivo CSV tenga las columnas correctas.

3. **"Memory error"**: Reduce el tamaño del dataset o usa menos modelos en `get_model_configs()`.

4. **"Import error"**: Verifica que todas las dependencias estén instaladas con `pip install -r requirements.txt`.

### Debugging

```python
# Habilitar logs detallados
import logging
logging.basicConfig(level=logging.DEBUG)

# Verificar estado del predictor
print(f"Modelo entrenado: {predictor.is_trained}")
print(f"Mejor modelo: {predictor.best_model_name}")
```

## Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles. 