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
from src.config import (
    FILE_TERENCICP_GAMES,
    FILE_TERENCICP_CATEGORIES,
    FILE_TERENCICP_TAGS,
    FILE_FRONKONGAMES,
    TAG_FREQ_THRESHOLD,
)


def load_terencicp() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los tres archivos del dataset terencicp."""
    raise NotImplementedError("Implementar en feature/data-pipeline")


def load_fronkongames() -> pd.DataFrame:
    """Carga el dataset de fronkongames y selecciona columnas relevantes."""
    raise NotImplementedError("Implementar en feature/data-pipeline")


def merge_datasets(
    df_games: pd.DataFrame,
    df_categories: pd.DataFrame,
    df_tags: pd.DataFrame,
    df_fronkon: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une los cuatro dataframes en un único dataset enriquecido.
    Estrategia:
      1. Pivot de categorías (formato largo → columnas binarias).
      2. Filtro y pivot de tags por tag_frequencies.
      3. Merge externo con fronkongames por AppID (how='left').
    """
    raise NotImplementedError("Implementar en feature/data-pipeline")


def load_all() -> pd.DataFrame:
    """Punto de entrada principal: carga y unifica todos los datasets."""
    df_games, df_categories, df_tags = load_terencicp()
    df_fronkon = load_fronkongames()
    return merge_datasets(df_games, df_categories, df_tags, df_fronkon)
