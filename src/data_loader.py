"""
data_loader.py
--------------
Carga y unificación de los dos datasets de Steam:
  - terencicp/steam-games-december-2023 (games_terencicp.csv, t-games-categories.csv, t-games-tags.csv)
  - fronkongames/steam-games-dataset    (games_fronkon.csv)

El merge externo se realiza por AppID para enriquecer el dataset original
con columnas faltantes: Developers, Publishers, Genres, Windows/Mac/Linux,
Supported languages, Achievements y Metacritic score.
"""


import pandas as pd

##Cambio 1
from pathlib import Path

from src.config import (
    FILE_TERENCICP_GAMES,
    FILE_TERENCICP_CATEGORIES,
    FILE_TERENCICP_TAGS,
    FILE_FRONKONGAMES,
    TAG_FREQ_THRESHOLD,
)

# Columnas del dataset fronkongames que aportan información pre-lanzamiento
# no disponible en el dataset terencicp
FRONKON_COLS_MERGE = [
    "AppID",
    "Developers",
    "Publishers",
    "Genres",
    "Windows",
    "Mac",
    "Linux",
    "Supported languages",
    "Achievements",
    "Metacritic score",
    "Required age",
]

#Cambio 2
def validate_dataset_files():
    """
    Verifica que todos los datasets requeridos existan antes de cargarlos.
    """

    dataset_paths = {
        "games_terencicp": FILE_TERENCICP_GAMES,
        "categories_terencicp": FILE_TERENCICP_CATEGORIES,
        "tags_terencicp": FILE_TERENCICP_TAGS,
        "games_fronkongames": FILE_FRONKONGAMES,
    }

    for dataset, path in dataset_paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"No se encontró el dataset '{dataset}'."
            )


def load_terencicp() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los tres archivos del dataset terencicp.

    Returns
    -------
    df_games : pd.DataFrame
        Tabla principal con atributos base de cada juego.
    df_categories : pd.DataFrame
        Tabla en formato largo (app_id, categories).
    df_tags : pd.DataFrame
        Tabla en formato largo (app_id, tags, tag_frequencies).
    """
    df_games = pd.read_csv(FILE_TERENCICP_GAMES)
    df_categories = pd.read_csv(FILE_TERENCICP_CATEGORIES)
    df_tags = pd.read_csv(FILE_TERENCICP_TAGS)
    return df_games, df_categories, df_tags


def load_fronkongames() -> pd.DataFrame:
    """
    Carga el dataset de fronkongames y selecciona columnas relevantes.

    Nota técnica: el CSV exportado por fronkongames incluye el índice de pandas
    como primera columna sin nombre, desplazando todas las columnas una posición
    a la derecha. Se corrige con index_col=0 y renombrando las columnas afectadas.

    Returns
    -------
    pd.DataFrame con columnas: AppID + FRONKON_COLS_MERGE
    """
    df = pd.read_csv(FILE_FRONKONGAMES, index_col=0)

    # Corregir el desplazamiento de columnas causado por el índice exportado:
    # la columna 'AppID' dentro del df contiene en realidad los nombres de juego,
    # ya que el AppID numérico real quedó absorbido como índice.
    df = df.rename(columns={
        "AppID": "Name",
        "Name": "Release date",
        "Release date": "Estimated owners",
    })
    df.index.name = "AppID"

    # Limpiar filas con AppID no numérico o infinito
    import numpy as np
    numeric_index = pd.to_numeric(df.index, errors="coerce")
    df = df[numeric_index.notna() & ~np.isinf(numeric_index)]
    df.index = df.index.astype(int)
    df = df.reset_index().copy()

    return df[FRONKON_COLS_MERGE]

def _pivot_categories(df_categories: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte la tabla de categorías de formato largo a columnas binarias.
    Prefija cada columna con 'cat_' para distinguirlas de otras features.
    """
    pivot = (
        df_categories.groupby(["app_id", "categories"])
        .size()
        .unstack(fill_value=0)
        .clip(upper=1)
    )
    pivot.columns = [
        "cat_" + c.lower().replace(" ", "_").replace("-", "_")
        for c in pivot.columns
    ]
    return pivot

def _pivot_tags(df_tags: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra tags por tag_frequencies y convierte a columnas binarias.
    Solo se conservan asociaciones tag-juego con al menos TAG_FREQ_THRESHOLD votos,
    eliminando tags asignados por muy pocos usuarios (probable ruido).
    Prefija cada columna con 'tag_'.
    """
    df_filtered = df_tags[df_tags["tag_frequencies"] >= TAG_FREQ_THRESHOLD]
    pivot = (
        df_filtered.groupby(["app_id", "tags"])
        .size()
        .unstack(fill_value=0)
        .clip(upper=1)
    )
    pivot.columns = [
        "tag_" + t.lower().replace(" ", "_").replace("-", "_")
        for t in pivot.columns
    ]
    return pivot

def merge_datasets(
    df_games: pd.DataFrame,
    df_categories: pd.DataFrame,
    df_tags: pd.DataFrame,
    df_fronkon: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une los cuatro dataframes en un único dataset enriquecido.

    Estrategia:
      1. Pivot de categorías (formato largo → columnas binarias cat_*).
      2. Filtro y pivot de tags por tag_frequencies (columnas binarias tag_*).
      3. Join de categorías y tags sobre games por app_id (left join).
      4. Merge externo con fronkongames por AppID (left join).

    El uso de how='left' garantiza que no se pierda ningún juego del dataset
    base aunque no tenga match en las otras tablas.

    Returns
    -------
    pd.DataFrame con todas las columnas integradas.
    """
    categories_pivot = _pivot_categories(df_categories)
    tags_pivot = _pivot_tags(df_tags)

    df = (
        df_games.set_index("app_id")
        .merge(categories_pivot, left_index=True, right_index=True, how="left")
        .merge(tags_pivot, left_index=True, right_index=True, how="left")
    )

    bin_cols = [c for c in df.columns if c.startswith("cat_") or c.startswith("tag_")]
    df[bin_cols] = df[bin_cols].fillna(0).astype(int)
    df = df.reset_index().copy()

    df = df.merge(
        df_fronkon,
        left_on="app_id",
        right_on="AppID",
        how="left",
    ).drop(columns=["AppID"])

    return df

def load_all() -> pd.DataFrame:
    """
    Punto de entrada principal: carga y unifica todos los datasets.

    Returns
    -------
    pd.DataFrame con ~61.000 juegos y todas las columnas integradas.
    """
    try:
        validate_dataset_files()

        df_games, df_categories, df_tags = load_terencicp()
        df_fronkon = load_fronkongames()

        return merge_datasets(df_games, df_categories, df_tags, df_fronkon)
    
    except FileNotFoundError as e:
        print(e)
        return pd.DataFrame()


def get_merge_stats(df: pd.DataFrame) -> dict:
    """
    Retorna estadísticas del merge para reportar en el notebook.

    Returns
    -------
    dict con: total_games, matched_games, match_rate, null_rates_by_col
    """
    fronkon_cols = [c for c in FRONKON_COLS_MERGE if c != "AppID" and c in df.columns]
    null_rates = {col: df[col].isna().mean() for col in fronkon_cols}
    matched = df[fronkon_cols[0]].notna().sum() if fronkon_cols else 0

    return {
        "total_games": len(df),
        "matched_games": matched,
        "match_rate": matched / len(df) if len(df) > 0 else 0,
        "null_rates_by_col": null_rates,
    }

# ---------------------------------------------------------------------------
# Ejecución directa para verificación rápida del pipeline de carga
# Uso: python -m src.data_loader
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Cargando datasets...")
    df = load_all()
    stats = get_merge_stats(df)
    print(f"Dataset integrado: {df.shape[0]:,} juegos, {df.shape[1]} columnas")
    print(f"Match con fronkongames: {stats['matched_games']:,} / {stats['total_games']:,} ({stats['match_rate']:.1%})")