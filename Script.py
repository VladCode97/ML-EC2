# %% [markdown]
# # Taller 2 MLOPS
# ### Integrantes:  
# Nombre: Luis Eduardo Bonilla Torres.
# 
# 1144200150
# 
# luisbtcreative@gmail.com
# 
# Mauricio Posada Palma 
# 
# 1001366381
#  
# mauriciop1palma@gmail.com

# %% [markdown]
# # Entendimiento de negocio:
# - Definir objetivo de negocio según lo que usted considere que puede ser de mayor utilidad según los datos elegidos.
# - Definir objetivo analítico. 

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import recall_score, classification_report, confusion_matrix, roc_auc_score
from typing import Dict, List, Tuple, Any

# %% [markdown]
# # Funciones de Carga y Exploración de Datos

# %%
def load_data(file_path: str = "/Users/luis/Documents/AI Master/mlops-course/MLOpsWorkshop2/data.csv") -> pd.DataFrame:
    """
    Carga los datos desde el archivo CSV
    
    Args:
        file_path: Ruta al archivo CSV
        
    Returns:
        DataFrame con los datos cargados
    """
    print(f"Cargando datos desde: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Datos cargados: {df.shape}")
    return df

def explore_data(df: pd.DataFrame) -> None:
    """
    Realiza exploración básica de los datos
    
    Args:
        df: DataFrame a explorar
    """
    print("=== EXPLORACIÓN DE DATOS ===")
    print(f"Shape: {df.shape}")
    
    print("\nInfo:")
    df.info()
    
    print("\nPrimeras filas:")
    print(df.head())
    
    print("\nDescripción estadística:")
    print(df.describe())
    
    print("\nDistribución de la variable objetivo:")
    print(df["Bankrupt?"].value_counts())

def get_top_features(df: pd.DataFrame, n_features: int = 5) -> List[str]:
    """
    Obtiene las n features más correlacionadas con la variable objetivo
    
    Args:
        df: DataFrame con los datos
        n_features: Número de features a seleccionar
        
    Returns:
        Lista con los nombres de las features más correlacionadas
    """
    corr_target = df.corr()["Bankrupt?"].abs().drop("Bankrupt?")
    top_features = corr_target.sort_values(ascending=False).head(n_features).index.tolist()
    print(f"Top {n_features} features seleccionadas: {top_features}")
    return top_features

# %% [markdown]
# # Funciones de Visualización

# %%
def plot_feature_distributions(df: pd.DataFrame, features: List[str]) -> None:
    """
    Genera histogramas y boxplots para las features especificadas
    
    Args:
        df: DataFrame con los datos
        features: Lista de nombres de features a visualizar
    """
    print("Generando visualizaciones de distribuciones...")
    
    for col in features:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histograma con KDE
        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f"Distribución de {col}")
        axes[0].set_xlabel(col)
        
        # Boxplot para detectar outliers
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f"Boxplot de {col}")
        
        plt.tight_layout()
        plt.show()

def plot_target_distribution(df: pd.DataFrame) -> None:
    """
    Visualiza la distribución de la variable objetivo
    
    Args:
        df: DataFrame con los datos
    """
    plt.figure(figsize=(5,4))
    sns.countplot(x="Bankrupt?", data=df)
    plt.title("Distribución de la clase objetivo")
    plt.xlabel("Bankrupt?")
    plt.ylabel("Número de empresas")
    plt.show()

def plot_correlation_analysis(df: pd.DataFrame, top_features: List[str]) -> None:
    """
    Genera análisis de correlaciones y visualizaciones
    
    Args:
        df: DataFrame con los datos
        top_features: Lista de features principales
    """
    print("Generando análisis de correlaciones...")
    
    # Pairplot para features principales
    sns.pairplot(
        df[top_features + ["Bankrupt?"]],
        hue="Bankrupt?",
        diag_kind="kde",
        plot_kws={"alpha":0.5}
    )
    plt.suptitle("Pairplot: Relaciones bivariantes de top features", y=1.02)
    plt.show()
    
    # Heatmap de correlaciones
    plt.figure(figsize=(14,12))
    sns.heatmap(
        df.corr(),
        cmap="coolwarm",
        center=0,
        linewidths=0.2
    )
    plt.title("Mapa de calor de correlaciones entre todas las variables")
    plt.show()
    
    # Scatterplot para dos variables más relevantes
    if len(top_features) >= 2:
        x_var, y_var = top_features[0], top_features[1]
        plt.figure(figsize=(6,6))
        sns.scatterplot(
            data=df,
            x=x_var,
            y=y_var,
            hue="Bankrupt?",
            alpha=0.6
        )
        plt.title(f"Scatter: {x_var} vs {y_var} por clase")
        plt.show()

# %% [markdown]
# # Funciones de Preprocesamiento

# %%
def preprocess_data(df: pd.DataFrame, target_column: str = "Bankrupt?", 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Realiza todo el preprocesamiento de datos
    
    Args:
        df: DataFrame con los datos
        target_column: Nombre de la columna objetivo
        test_size: Proporción del conjunto de test
        random_state: Semilla para reproducibilidad
        
    Returns:
        X_train_bal, X_test, y_train_bal, y_test
    """
    print("=== PREPROCESAMIENTO DE DATOS ===")
    
    # Verificar que existe la columna objetivo
    assert target_column in df.columns, f"Columna {target_column} no encontrada. Columnas disponibles: {df.columns.tolist()}"
    
    # 1. Limpiar nombres de columnas
    df.columns = df.columns.str.strip()
    
    # 2. Separación de X e y
    X = df.drop(target_column, axis=1)
    y = df[target_column].astype(int)
    print(f"Features: {X.shape[1]}, Muestras: {X.shape[0]}")
    
    # 3. Reemplazar infinitos por NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # 4. Imputación de valores nulos usando KNNImputer
    print("Aplicando imputación KNN...")
    imputer = KNNImputer(n_neighbors=5)
    X_imputed_arr = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed_arr, columns=X.columns)
    
    # 5. Escalado de características
    print("Aplicando escalado...")
    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(X_imputed)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=X.columns)
    
    # 6. División train/test
    print("Dividiendo en train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 7. Balanceo de clases usando SMOTE
    print("Aplicando balanceo SMOTE...")
    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    # Mostrar distribuciones
    print("Distribución original (train):")
    print(y_train.value_counts())
    print("\nDistribución después de SMOTE:")
    print(pd.Series(y_train_bal).value_counts())
    
    return X_train_bal, X_test, y_train_bal, y_test

# %% [markdown]
# # Funciones de Entrenamiento de Modelos

# %%
def  get_model_configurations() -> Dict[str, Dict[str, Any]]:
    """
    Define las configuraciones de los modelos a entrenar
    
    Returns:
        Diccionario con configuraciones de modelos
    """
    return {
        'LogisticRegression': {
            'model': LogisticRegression(solver='liblinear'),
            'params': {'C': [0.01, 0.1, 1, 10]}
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': [3, 5, 7]}
        },
        'SVM': {
            'model': SVC(probability=True),
            'params': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [100, 200], 'max_depth': [5, 10]}
        },
        'XGBoost': {
            'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'params': {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
        }
    }

def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Entrena todos los modelos usando GridSearchCV
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        
    Returns:
        results: Diccionario con scores de validación cruzada
        best_models: Diccionario con los mejores modelos
    """
    print("=== ENTRENAMIENTO DE MODELOS ===")
    
    model_configs = get_model_configurations()
    results = {}
    best_models = {}
    
    for name, config in model_configs.items():
        print(f"Entrenando {name}...")
        
        model = GridSearchCV(
            config['model'], 
            config['params'], 
            cv=5, 
            scoring='recall', 
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        results[name] = model.best_score_
        best_models[name] = model.best_estimator_
        
        print(f"  {name} - Mejor score CV: {model.best_score_:.3f}")
    
    return results, best_models

# %% [markdown]
# # Funciones de Evaluación

# %%
def evaluate_models(best_models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
    """
    Evalúa todos los modelos en el conjunto de test
    
    Args:
        best_models: Diccionario con los mejores modelos
        X_test: Features de test
        y_test: Target de test
        
    Returns:
        Diccionario con métricas de evaluación para cada modelo
    """
    print("=== EVALUACIÓN EN TEST ===")
    
    evaluation_results = {}
    
    for name, model in best_models.items():
        print(f"\nEvaluando {name}...")
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        
        evaluation_results[name] = {
            'recall': recall,
            'auc': auc,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"  Recall: {recall:.3f}")
        if auc is not None:
            print(f"  AUC: {auc:.3f}")
    
    return evaluation_results

def get_best_model(best_models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> str:
    """
    Determina el mejor modelo según recall en test
    
    Args:
        best_models: Diccionario con los mejores modelos
        X_test: Features de test
        y_test: Target de test
        
    Returns:
        Nombre del mejor modelo
    """
    best_model_name = max(
        best_models.keys(), 
        key=lambda name: recall_score(y_test, best_models[name].predict(X_test))
    )
    print(f"\n🏆 Mejor modelo según recall en test: {best_model_name}")
    return best_model_name

def print_detailed_evaluation(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> None:
    """
    Imprime evaluación detallada de un modelo específico
    
    Args:
        model: Modelo a evaluar
        X_test: Features de test
        y_test: Target de test
        model_name: Nombre del modelo
    """
    print(f"\n=== EVALUACIÓN DETALLADA ({model_name}) ===")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"AUC: {roc_auc_score(y_test, y_prob):.3f}")

# %% [markdown]
# # Función Principal del Pipeline

# %%
def run_bankruptcy_pipeline(data_path: str = "/Users/luis/Documents/AI Master/mlops-course/MLOpsWorkshop2/data.csv",
                           show_plots: bool = False) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Ejecuta el pipeline completo de predicción de bancarrota
    
    Args:
        data_path: Ruta al archivo de datos
        show_plots: Si mostrar las visualizaciones
        
    Returns:
        best_model_name: Nombre del mejor modelo
        results: Resultados de entrenamiento
        evaluation_results: Resultados de evaluación
    """
    print("🚀 INICIANDO PIPELINE DE PREDICCIÓN DE BANCARROTA")
    print("=" * 60)
    
    # 1. Cargar datos
    df = load_data(data_path)
    
    # 2. Exploración de datos
    explore_data(df)
    
    # 3. Obtener features principales
    top_features = get_top_features(df)
    
    # 4. Visualizaciones (opcional)
    if show_plots:
        plot_feature_distributions(df, top_features)
        plot_target_distribution(df)
        plot_correlation_analysis(df, top_features)
    
    # 5. Preprocesamiento
    X_train_bal, X_test, y_train_bal, y_test = preprocess_data(df)
    
    # 6. Entrenamiento de modelos
    results, best_models = train_models(X_train_bal, y_train_bal)
    
    # 7. Evaluación
    evaluation_results = evaluate_models(best_models, X_test, y_test)
    
    # 8. Determinar mejor modelo
    best_model_name = get_best_model(best_models, X_test, y_test)
    
    # 9. Evaluación detallada del mejor modelo
    best_model = best_models[best_model_name]
    print_detailed_evaluation(best_model, X_test, y_test, best_model_name)
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    
    return best_model_name, results, evaluation_results

# %% [markdown]
# # Función para Predicciones

# %%
def predict_bankruptcy(data: Dict[str, float], best_model: Any, 
                      preprocessor_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Realiza predicción de bancarrota para nuevos datos
    
    Args:
        data: Diccionario con features de la empresa
        best_model: Mejor modelo entrenado
        preprocessor_components: Componentes del preprocesador (imputer, scaler)
        
    Returns:
        Diccionario con predicción y probabilidad
    """
    # Convertir a DataFrame
    df = pd.DataFrame([data])
    
    # Aplicar preprocesamiento
    df_processed = df.replace([np.inf, -np.inf], np.nan)
    df_imputed = pd.DataFrame(
        preprocessor_components['imputer'].transform(df_processed),
        columns=df_processed.columns
    )
    df_scaled = pd.DataFrame(
        preprocessor_components['scaler'].transform(df_imputed),
        columns=df_imputed.columns
    )
    
    # Realizar predicción
    prediction = best_model.predict(df_scaled)[0]
    probability = best_model.predict_proba(df_scaled)[0, 1]
    
    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'risk_level': 'ALTO' if probability > 0.7 else 'MEDIO' if probability > 0.3 else 'BAJO'
    }

# %% [markdown]
# # Función para uso en FastAPI

# %%
def best_model_predict() -> str:
    """
    Función para obtener el nombre del mejor modelo
    Útil para integración con FastAPI
    
    Returns:
        Nombre del mejor modelo
    """
    # Ejecutar pipeline si no se ha ejecutado
    if 'best_model_name' not in globals():
        global best_model_name, results, evaluation_results
        best_model_name, results, evaluation_results = run_bankruptcy_pipeline(show_plots=False)
    
    return best_model_name

# %% [markdown]
# # Ejecución del Pipeline

# %%
if __name__ == "__main__":
    # Ejecutar pipeline completo
    best_model_name, results, evaluation_results = run_bankruptcy_pipeline()
    
    # Ejemplo de predicción
    print("\n" + "=" * 60)
    print("📊 EJEMPLO DE PREDICCIÓN")
    
    # Datos de ejemplo
    sample_company = {
        "Net Income to Total Assets": 0.1,
        "ROA(A) before interest and % after tax": 0.05,
        "ROA(B) before interest and depreciation after tax": 0.03,
        "ROA(C) before interest and depreciation before interest": 0.02,
        "Net worth/Assets": 0.8
    }
    
    # Obtener componentes del preprocesador (necesario para predicciones)
    # Nota: En una implementación real, estos se guardarían durante el entrenamiento
    print("⚠️  Nota: Para predicciones reales, necesitas guardar los componentes del preprocesador")
    print(f"📈 Mejor modelo: {best_model_name}")

# %% [markdown]
# ## 5.1. Métrica clave: Recall
# 
# **Definición:**
# 
# $$
# \mathrm{Recall} \;=\; \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}
# $$
# 
# donde  
# - **TP** = verdaderos positivos (empresas en bancarrota correctamente identificadas)  
# - **FN** = falsos negativos (empresas en bancarrota que no detectamos)  
# 
# **Justificación de negocio:**
# 
# - Un **falso negativo** (pasar por alto una quiebra real) puede causar pérdidas millonarias, impagos y daño reputacional.  
# - Preferimos tolerar **falsas alarmas** (falsos positivos) antes que perder un caso real de bancarrota.
# 
# **Contexto de desequilibrio:**
# 
# - Solo el ~3 % de las empresas quiebran → el modelo tiende a ignorar casos positivos.  
# - **Maximizar recall** garantiza capturar la mayoría de riesgos.
# 


