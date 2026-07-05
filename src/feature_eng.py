"""
feature_eng.py
--------------
Limpieza, transformación y construcción de features para el modelo.

Responsabilidades:
  - Exclusión de variables post-lanzamiento.
  - Filtro de calidad de la variable objetivo (reseñas mínimas).
  - Construcción de la variable objetivo multiclase (3 0 4 niveles Steam).
  - One-Hot Encoding de géneros, categorías y tags.
  - Extracción de release_year desde release_date.
"""

import pandas as pd
from src.config import (
    MIN_REVIEWS,
    THRESH_OVERWHELM,
    THRESH_POSITIVE,
    THRESH_MIXED_LOW,
    MIN_REVIEWS_OVERWHELM,
    CLASS_LABELS,
    N_CLASSES,
)

# Columnas post-lanzamiento que nunca se usan como features
POST_LAUNCH_COLS = [
    "positive",
    "negative",
    "min_owners",
    "max_owners",
    "hltb_single",
    "Positive",
    "Negative",
    "Score rank",
    "User score",
    "Recommendations",
    "Peak CCU",
    "Average playtime forever",
    "Average playtime two weeks",
    "Median playtime forever",
    "Median playtime two weeks",
    "Estimated owners",
]


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula positive_ratio y construye la variable objetivo multiclase
    siguiendo la lógica oficial de Steam, en 3 o 4 clases según N_CLASSES.

    4 clases:
        Negative            (ratio < 40%)
        Mixed               (40% ≤ ratio < 70%)
        Positive            (70% ≤ ratio < 95% o < 500 reseñas)
        Overwhelmingly Positive (≥ 95% y ≥ 500 reseñas)

    3 clases (fusiona Overwhelmingly Positive con Positive):
        Negative            (ratio < 40%)
        Mixed               (40% ≤ ratio < 70%)
        Positive            (≥ 70%)
    """
    df = df.copy()
    df["positive_ratio"] = df["positive"] / (df["positive"] + df["negative"])

    def classify_4(row: pd.Series) -> str:
        ratio = row["positive_ratio"]
        total = row["total_reviews"]
        if ratio >= THRESH_OVERWHELM and total >= MIN_REVIEWS_OVERWHELM:
            return "Overwhelmingly Positive"
        elif ratio >= THRESH_POSITIVE:
            return "Positive"
        elif ratio >= THRESH_MIXED_LOW:
            return "Mixed"
        else:
            return "Negative"

    def classify_3(row: pd.Series) -> str:
        ratio = row["positive_ratio"]
        if ratio >= THRESH_POSITIVE:
            return "Positive"
        elif ratio >= THRESH_MIXED_LOW:
            return "Mixed"
        else:
            return "Negative"

    classify_fn = classify_4 if N_CLASSES == 4 else classify_3
    df["success_level"] = df.apply(classify_fn, axis=1)
    df["success_level"] = pd.Categorical(
        df["success_level"], categories=CLASS_LABELS, ordered=True
    )

    print(f"build_target ({N_CLASSES} clases): distribución:")
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


def encode_fronkon_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Codifica columnas nuevas provenientes del merge con fronkongames:
      - Genres      : texto con géneros separados por coma → One-Hot Encoding
      - Developers  : nombre del desarrollador → One-Hot Encoding (top frecuentes)
      - Windows/Mac/Linux : booleanos → convertir a int (0/1)
      - Required age: numérica → usar directamente
      - Achievements: numérica → usar directamente
      - Metacritic score: numérica → usar directamente (muchos NaN esperados)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con columnas de fronkongames ya mergeadas.

    Returns
    -------
    df : pd.DataFrame con nuevas columnas codificadas
    fronkon_features : list[str] — nombres de las nuevas features generadas
    """
    df = df.copy()
    fronkon_features = []

    # Plataformas: booleanos → int
    for col in ["Windows", "Mac", "Linux"]:
        if col in df.columns:
            new_col = f"platform_{col.lower()}"
            df[new_col] = df[col].fillna(False).astype(int)
            fronkon_features.append(new_col)

    # Variables numéricas directas
    for col, new_col in [
        ("Required age", "required_age"),
        ("Achievements", "achievements"),
        ("Metacritic score", "metacritic_score"),
    ]:
        if col in df.columns:
            df[new_col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            fronkon_features.append(new_col)

    # Genres: texto separado por coma → One-Hot Encoding
    if "Genres" in df.columns:
        genres_dummies = (
            df["Genres"]
            .fillna("")
            .str.get_dummies(sep=",")
        )
        genres_dummies.columns = [
            "genre_" + g.strip().lower().replace(" ", "_")
            for g in genres_dummies.columns
        ]
        df = pd.concat([df, genres_dummies], axis=1)
        fronkon_features.extend(genres_dummies.columns.tolist())

    # Developers: One-Hot Encoding solo para los top 50 más frecuentes
    # (evita explosión de dimensionalidad con miles de estudios distintos)
    if "Developers" in df.columns:
        top_devs = df["Developers"].value_counts().head(50).index
        for dev in top_devs:
            col_name = "dev_" + dev.lower().replace(" ", "_").replace("-", "_")[:30]
            df[col_name] = (df["Developers"] == dev).astype(int)
            fronkon_features.append(col_name)

    print(f"encode_fronkon_features: {len(fronkon_features)} features nuevas generadas")
    return df, fronkon_features


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    """
    Pipeline completo de feature engineering. Orquesta todos los pasos
    anteriores en el orden correcto y retorna el dataset listo para modelar.

    Pasos:
      1. Filtro de reseñas mínimas
      2. Construcción de la variable objetivo multiclase
      3. Extracción de features temporales (release_year)
      4. Codificación de categorías y tags (cat_*, tag_*)
      5. Codificación de features de fronkongames (géneros, plataformas, etc.)
      6. Exclusión de columnas post-lanzamiento del feature set

    Parameters
    ----------
    df : pd.DataFrame
        Dataset integrado proveniente de data_loader.load_all().

    Returns
    -------
    df_model : pd.DataFrame — dataset listo para train/test split
    features  : list[str]  — lista de columnas a usar como features
    target    : str        — nombre de la columna objetivo ('success_level')
    """
    TARGET = "success_level"

    # Paso 1-3: filtros y target
    df = filter_min_reviews(df)
    df = build_target(df)
    df = extract_temporal_features(df)

    # Paso 4-5: encoding
    df, cat_tag_feats = encode_categories_tags(df)
    df, fronkon_feats = encode_fronkon_features(df)

    # Features numéricas base (pre-lanzamiento)
    base_features = ["price", "release_year"]

    # Feature set completo
    features = base_features + cat_tag_feats + fronkon_feats

    # Excluir columnas post-lanzamiento del dataframe de modelado
    cols_to_drop = [c for c in POST_LAUNCH_COLS if c in df.columns]
    df_model = df[features + [TARGET]].dropna(subset=["release_year"])

    print(f"\nprepare_dataset completado:")
    print(f"  Filas: {len(df_model):,}")
    print(f"  Features totales: {len(features)}")
    print(f"  Target: '{TARGET}'")

    return df_model, features, TARGET