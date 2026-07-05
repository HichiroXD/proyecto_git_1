"""
model.py
--------
Definición y entrenamiento de los modelos de clasificación multiclase.

Incluye:
  - CumulativeImportanceSelector: transformer sklearn personalizado que
    selecciona features por importancia acumulada (Random Forest interno),
    encapsulado en el Pipeline para evitar data leakage.
  - build_pipeline_dt(): Pipeline con Decision Tree multiclase.
  - build_pipeline_rf(): Pipeline con Random Forest multiclase.
  - train_and_evaluate(): entrenamiento con validación cruzada estratificada.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from src.config import (
    CUM_IMP_THRESHOLD,
    TEST_SIZE,
    RANDOM_STATE,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_SPLIT,
    RF_MIN_SAMPLES_LEAF,
)


class CumulativeImportanceSelector(BaseEstimator, TransformerMixin):
    """
    Transformer sklearn que selecciona el menor subconjunto de features
    cuya importancia acumulada (medida con un Random Forest interno)
    alcanza cum_threshold.

    Al estar dentro de un Pipeline, se reajusta solo con los datos de
    entrenamiento de cada fold/split — sin fuga de datos hacia el test set.

    Parameters
    ----------
    cum_threshold : float
        Porcentaje de importancia acumulada mínima (default: 0.90).
    n_estimators : int
        Número de árboles del Random Forest interno.
    max_depth : int
        Profundidad máxima del Random Forest interno.
    random_state : int
        Semilla aleatoria para reproducibilidad.
    """

    def __init__(
        self,
        cum_threshold: float = CUM_IMP_THRESHOLD,
        n_estimators: int = RF_N_ESTIMATORS,
        max_depth: int = RF_MAX_DEPTH,
        random_state: int = RANDOM_STATE,
    ):
        self.cum_threshold = cum_threshold
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        importances = pd.Series(
            rf.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        cum = importances.cumsum()
        n_selected = (cum <= self.cum_threshold).sum() + 1
        self.selected_features_ = importances.head(n_selected).index.tolist()
        self.importances_ = importances
        return self

    def transform(self, X):
        return X[self.selected_features_]


def build_pipeline_dt() -> Pipeline:
    """
    Construye el Pipeline de Decision Tree multiclase.

    El Pipeline encadena:
      1. CumulativeImportanceSelector — selección de features sin data leakage
      2. DecisionTreeClassifier       — clasificador multiclase

    Los hiperparámetros de control de overfitting (max_depth, min_samples_*)
    se definen en config.py para centralizar su gestión.

    Returns
    -------
    Pipeline sklearn listo para .fit() y .predict()
    """
    return Pipeline([
        ("selector", CumulativeImportanceSelector(cum_threshold=CUM_IMP_THRESHOLD)),
        ("clf", DecisionTreeClassifier(
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])


def build_pipeline_rf() -> Pipeline:
    """
    Construye el Pipeline de Random Forest multiclase.

    El Pipeline encadena:
      1. CumulativeImportanceSelector — selección de features sin data leakage
      2. RandomForestClassifier       — ensamble de árboles multiclase

    Returns
    -------
    Pipeline sklearn listo para .fit() y .predict()
    """
    return Pipeline([
        ("selector", CumulativeImportanceSelector(cum_threshold=CUM_IMP_THRESHOLD)),
        ("clf", RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])


def split_data(
    df: pd.DataFrame, features: list[str], target: str
) -> tuple:
    """
    División estratificada 80/20 en conjuntos de entrenamiento y prueba.

    Se usa stratify=y para preservar la proporción de cada clase en ambos
    conjuntos — especialmente importante en clasificación multiclase donde
    alguna clase puede tener pocas muestras (ej. Overwhelmingly Positive).

    Parameters
    ----------
    df       : pd.DataFrame — dataset completo de modelado
    features : list[str]   — columnas a usar como features
    target   : str         — nombre de la columna objetivo

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"split_data: train={len(X_train):,} | test={len(X_test):,}")
    print("Distribución en train:")
    print(y_train.value_counts().sort_index())

    return X_train, X_test, y_train, y_test


def cross_validate_pipeline(
    pipeline: Pipeline, X_train, y_train, cv: int = 5
) -> dict:
    """
    Validación cruzada estratificada sobre el conjunto de entrenamiento.

    Usa macro-F1 como métrica principal por ser más robusta ante el
    desbalance de clases que accuracy — especialmente relevante aquí
    ya que 'Overwhelmingly Positive' tiene muchas menos muestras que
    'Mostly Positive'.

    Parameters
    ----------
    pipeline : Pipeline sklearn ya construido (sin entrenar)
    X_train  : pd.DataFrame — features de entrenamiento
    y_train  : pd.Series    — target de entrenamiento
    cv       : int          — número de folds (default: 5)

    Returns
    -------
    dict con: mean_f1, std_f1, scores (array por fold)
    """
    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
    )

    result = {
        "mean_f1": scores.mean(),
        "std_f1": scores.std(),
        "scores": scores,
    }

    print(f"cross_validate_pipeline ({cv}-fold CV):")
    print(f"  macro-F1: {result['mean_f1']:.3f} ± {result['std_f1']:.3f}")
    print(f"  folds: {scores.round(3)}")

    return result


# ---------------------------------------------------------------------------
# Ejecución directa para verificación rápida del pipeline de modelado
# Uso: python -m src.model
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import load_all
    from src.feature_eng import prepare_dataset

    print("Cargando datos...")
    df_datasets = load_all()

    print("Preparando features...")
    df_model, features, target = prepare_dataset(df_datasets)

    print("Dividiendo datos...")
    X_train, X_test, y_train, y_test = split_data(df_model, features, target)

    print("\nEntrenando Decision Tree...")
    pipeline_dt = build_pipeline_dt()
    pipeline_dt.fit(X_train, y_train)
    print(f"  Train acc: {pipeline_dt.score(X_train, y_train):.3f}")
    print(f"  Test  acc: {pipeline_dt.score(X_test, y_test):.3f}")

    print("\nEntrenando Random Forest...")
    pipeline_rf = build_pipeline_rf()
    pipeline_rf.fit(X_train, y_train)
    print(f"  Train acc: {pipeline_rf.score(X_train, y_train):.3f}")
    print(f"  Test  acc: {pipeline_rf.score(X_test, y_test):.3f}")