"""
feature_eng.py
--------------
Limpieza, transformación y construcción de features para el modelo.

Responsabilidades:
  - Exclusión de variables post-lanzamiento.
  - Filtro de calidad de la variable objetivo (reseñas mínimas).
  - Construcción de la variable objetivo multiclase (5 niveles Steam).
  - One-Hot Encoding de géneros, categorías y tags.
  - Extracción de release_year desde release_date.
"""

import pandas as pd
from src.config import (
    MIN_REVIEWS,
    THRESH_OVERWHELM,
    THRESH_VERY_POSITIVE,
    THRESH_MOSTLY,
    THRESH_MIXED_LOW,
    MIN_REVIEWS_OVERWHELM,
    MIN_REVIEWS_VERY,
    CLASS_LABELS,
)

# Columnas post-lanzamiento que nunca se usan como features
POST_LAUNCH_COLS = [
    "Positive", "Negative", "Score rank", "User score",
    "Recommendations", "Peak CCU",
    "Average playtime forever", "Average playtime two weeks",
    "Median playtime forever", "Median playtime two weeks",
    "Estimated owners",
]


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula positive_ratio y construye la variable objetivo multiclase
    siguiendo las categorías oficiales de Steam.

    Clases (de menor a mayor):
        0 - Negative          (ratio < 40%)
        1 - Mixed             (40% ≤ ratio < 70%)
        2 - Mostly Positive   (70% ≤ ratio < 80%)
        3 - Very Positive     (≥ 80% y ≥ 50 reseñas)
        4 - Overwhelmingly Positive (≥ 95% y ≥ 500 reseñas)
    """
    raise NotImplementedError("Implementar en feature/data-pipeline")


def filter_min_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina juegos con menos de MIN_REVIEWS reseñas totales."""
    raise NotImplementedError("Implementar en feature/data-pipeline")


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae release_year desde la columna de fecha de lanzamiento."""
    raise NotImplementedError("Implementar en feature/data-pipeline")


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Aplica One-Hot Encoding a géneros, categorías y tags.
    Retorna el dataframe transformado y la lista de features resultantes.
    """
    raise NotImplementedError("Implementar en feature/data-pipeline")


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    """
    Pipeline completo de feature engineering.
    Retorna (df_model, feature_list, target_col).
    """
    raise NotImplementedError("Implementar en feature/data-pipeline")
