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
        raise NotImplementedError("Implementar en feature/modeling")

    def transform(self, X):
        raise NotImplementedError("Implementar en feature/modeling")


def build_pipeline_dt() -> Pipeline:
    """Pipeline: CumulativeImportanceSelector + DecisionTreeClassifier."""
    raise NotImplementedError("Implementar en feature/modeling")


def build_pipeline_rf() -> Pipeline:
    """Pipeline: CumulativeImportanceSelector + RandomForestClassifier."""
    raise NotImplementedError("Implementar en feature/modeling")


def split_data(
    df: pd.DataFrame, features: list[str], target: str
) -> tuple:
    """División estratificada 80/20 en train y test."""
    raise NotImplementedError("Implementar en feature/modeling")


def cross_validate_pipeline(pipeline: Pipeline, X_train, y_train) -> dict:
    """Validación cruzada estratificada 5-fold. Retorna métricas resumidas."""
    raise NotImplementedError("Implementar en feature/modeling")
