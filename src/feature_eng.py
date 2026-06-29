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

    Clases (de menor a mayor éxito):
        0 - Negative            (ratio < 40%)
        1 - Mixed               (40% ≤ ratio < 70%)
        2 - Mostly Positive     (70% ≤ ratio < 80%)
        3 - Very Positive       (≥ 80% y ≥ MIN_REVIEWS_VERY reseñas)
        4 - Overwhelmingly Positive (≥ 95% y ≥ MIN_REVIEWS_OVERWHELM reseñas)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con columnas 'positive' y 'negative' (tras filter_min_reviews).

    Returns
    -------
    pd.DataFrame con columnas adicionales: 'positive_ratio' y 'success_level'.
    """
    df = df.copy()
    df["positive_ratio"] = df["positive"] / (df["positive"] + df["negative"])

    def classify(row: pd.Series) -> str:
        ratio = row["positive_ratio"]
        total = row["total_reviews"]

        if ratio >= THRESH_OVERWHELM and total >= MIN_REVIEWS_OVERWHELM:
            return "Overwhelmingly Positive"
        elif ratio >= THRESH_VERY_POSITIVE and total >= MIN_REVIEWS_VERY:
            return "Very Positive"
        elif ratio >= THRESH_MOSTLY:
            return "Mostly Positive"
        elif ratio >= THRESH_MIXED_LOW:
            return "Mixed"
        else:
            return "Negative"

    df["success_level"] = df.apply(classify, axis=1)

    # Convertir a categórica ordenada para que los modelos respeten el orden
    df["success_level"] = pd.Categorical(
        df["success_level"],
        categories=CLASS_LABELS,
        ordered=True,
    )

    print("build_target: distribución de clases:")
    print(df["success_level"].value_counts().sort_index())
    return df


def filter_min_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina juegos con menos de MIN_REVIEWS reseñas totales.

    Justificación: juegos con muy pocas reseñas producen un positive_ratio
    estadísticamente poco confiable (valores extremos por azar). Por ejemplo,
    un juego con 4 reseñas positivas y 1 negativa ya alcanza el 80% exacto,
    sin que eso refleje necesariamente una recepción genuinamente positiva.
    Este filtro mejora la calidad de la variable objetivo.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset integrado con columnas 'positive' y 'negative'.

    Returns
    -------
    pd.DataFrame filtrado, sin juegos de bajo volumen de reseñas.
    """
    df = df.copy()
    df["total_reviews"] = df["positive"] + df["negative"]
    n_before = len(df)
    df = df[df["total_reviews"] >= MIN_REVIEWS].copy()
    n_after = len(df)
    print(f"filter_min_reviews: {n_before:,} → {n_after:,} juegos "
          f"(eliminados: {n_before - n_after:,}, {(n_before-n_after)/n_before:.1%})")
    return df


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae el año de lanzamiento desde la columna de fecha.

    El año de lanzamiento es una feature pre-lanzamiento válida (el desarrollador
    decide cuándo lanzar el juego). Sin embargo, presenta una correlación
    parcial con la variable objetivo que refleja un sesgo de captura del dataset
    (los juegos más recientes tienen menos reseñas acumuladas). Esta limitación
    se documenta en el README.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con columna 'release_date' en formato texto (ej: 'Oct 12, 2017').

    Returns
    -------
    pd.DataFrame con columna adicional 'release_year' (int).
    """
    df = df.copy()
    df["release_year"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    ).dt.year.astype("Int64")

    null_years = df["release_year"].isna().sum()
    if null_years > 0:
        print(f"extract_temporal_features: {null_years} juegos sin año detectado (se excluirán en prepare_dataset)")

    return df


def encode_categories_tags(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Selecciona y retorna las columnas binarias cat_* y tag_* generadas
    por data_loader._pivot_categories() y data_loader._pivot_tags().

    Estas columnas ya están en formato binario (0/1) desde la carga,
    por lo que no requieren One-Hot Encoding adicional.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset integrado con columnas cat_* y tag_*.

    Returns
    -------
    df : pd.DataFrame (sin modificar)
    cat_tag_features : list[str] — nombres de columnas cat_* y tag_*
    """
    cat_tag_features = [
        c for c in df.columns
        if c.startswith("cat_") or c.startswith("tag_")
    ]
    print(f"encode_categories_tags: {len(cat_tag_features)} features "
          f"({sum(c.startswith('cat_') for c in cat_tag_features)} cat, "
          f"{sum(c.startswith('tag_') for c in cat_tag_features)} tag)")
    return df, cat_tag_features


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    """
    Pipeline completo de feature engineering.
    Retorna (df_model, feature_list, target_col).
    """
    raise NotImplementedError("Implementar en feature/data-pipeline")
