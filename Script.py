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

def load_data(file_path: str = "data.csv") -> pd.DataFrame:
    print(f"Cargando datos desde: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Datos cargados: {df.shape}")
    return df

def explore_data(df: pd.DataFrame) -> None:
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
    corr_target = df.corr()["Bankrupt?"].abs().drop("Bankrupt?")
    top_features = corr_target.sort_values(ascending=False).head(n_features).index.tolist()
    print(f"Top {n_features} features seleccionadas: {top_features}")
    return top_features

def plot_feature_distributions(df: pd.DataFrame, features: List[str]) -> None:

    print("Generando visualizaciones de distribuciones...")
    
    for col in features:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f"Distribución de {col}")
        axes[0].set_xlabel(col)
        
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f"Boxplot de {col}")
        
        plt.tight_layout()
        plt.show()

def plot_target_distribution(df: pd.DataFrame) -> None:

    plt.figure(figsize=(5,4))
    sns.countplot(x="Bankrupt?", data=df)
    plt.title("Distribución de la clase objetivo")
    plt.xlabel("Bankrupt?")
    plt.ylabel("Número de empresas")
    plt.show()

def plot_correlation_analysis(df: pd.DataFrame, top_features: List[str]) -> None:

    print("Generando análisis de correlaciones...")
    
    sns.pairplot(
        df[top_features + ["Bankrupt?"]],
        hue="Bankrupt?",
        diag_kind="kde",
        plot_kws={"alpha":0.5}
    )
    plt.suptitle("Pairplot: Relaciones bivariantes de top features", y=1.02)
    plt.show()
    
    plt.figure(figsize=(14,12))
    sns.heatmap(
        df.corr(),
        cmap="coolwarm",
        center=0,
        linewidths=0.2
    )
    plt.title("Mapa de calor de correlaciones entre todas las variables")
    plt.show()
    
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


def preprocess_data(df: pd.DataFrame, target_column: str = "Bankrupt?", 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    print("=== PREPROCESAMIENTO DE DATOS ===")
    
    assert target_column in df.columns, f"Columna {target_column} no encontrada. Columnas disponibles: {df.columns.tolist()}"
    
    df.columns = df.columns.str.strip()
    
    X = df.drop(target_column, axis=1)
    y = df[target_column].astype(int)
    print(f"Features: {X.shape[1]}, Muestras: {X.shape[0]}")
    
    X = X.replace([np.inf, -np.inf], np.nan)
    
    print("Aplicando imputación KNN...")
    imputer = KNNImputer(n_neighbors=5)
    X_imputed_arr = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed_arr, columns=X.columns)
    
    print("Aplicando escalado...")
    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(X_imputed)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=X.columns)
    
    print("Dividiendo en train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print("Aplicando balanceo SMOTE...")
    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    print("Distribución original (train):")
    print(y_train.value_counts())
    print("\nDistribución después de SMOTE:")
    print(pd.Series(y_train_bal).value_counts())
    
    return X_train_bal, X_test, y_train_bal, y_test

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

def evaluate_models(best_models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
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
    best_model_name = max(
        best_models.keys(), 
        key=lambda name: recall_score(y_test, best_models[name].predict(X_test))
    )
    print(f"\n🏆 Mejor modelo según recall en test: {best_model_name}")
    return best_model_name

def print_detailed_evaluation(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> None:
    print(f"\n=== EVALUACIÓN DETALLADA ({model_name}) ===")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"AUC: {roc_auc_score(y_test, y_prob):.3f}")

def run_bankruptcy_pipeline(data_path: str = "data.csv",
                           show_plots: bool = False) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    print("🚀 INICIANDO PIPELINE DE PREDICCIÓN DE BANCARROTA")
    print("=" * 60)
    df = load_data(data_path)
    explore_data(df)
    top_features = get_top_features(df)
    if show_plots:
        plot_feature_distributions(df, top_features)
        plot_target_distribution(df)
        plot_correlation_analysis(df, top_features)

    X_train_bal, X_test, y_train_bal, y_test = preprocess_data(df)

    results, best_models = train_models(X_train_bal, y_train_bal)

    evaluation_results = evaluate_models(best_models, X_test, y_test)

    best_model_name = get_best_model(best_models, X_test, y_test)

    best_model = best_models[best_model_name]
    print_detailed_evaluation(best_model, X_test, y_test, best_model_name)
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    
    return best_model_name, results, evaluation_results

def predict_bankruptcy(data: Dict[str, float], best_model: Any, 
                      preprocessor_components: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.DataFrame([data])
    df_processed = df.replace([np.inf, -np.inf], np.nan)
    df_imputed = pd.DataFrame(
        preprocessor_components['imputer'].transform(df_processed),
        columns=df_processed.columns
    )
    df_scaled = pd.DataFrame(
        preprocessor_components['scaler'].transform(df_imputed),
        columns=df_imputed.columns
    )
    prediction = best_model.predict(df_scaled)[0]
    probability = best_model.predict_proba(df_scaled)[0, 1]
    
    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'risk_level': 'ALTO' if probability > 0.7 else 'MEDIO' if probability > 0.3 else 'BAJO'
    }

# %%
def best_model_predict() -> str:
    if 'best_model_name' not in globals():
        global best_model_name, results, evaluation_results
        best_model_name, results, evaluation_results = run_bankruptcy_pipeline(show_plots=False)
    
    return best_model_name
