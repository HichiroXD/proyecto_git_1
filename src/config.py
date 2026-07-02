"""
config.py
---------
Constantes globales, rutas de archivos y umbrales del proyecto.
Centralizar aquí evita hardcodear valores en múltiples módulos.
"""

from pathlib import Path

# Raíz del proyecto (dos niveles arriba de src/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Rutas de datos
DATA_DATASETS_DIR = ROOT_DIR / "data" / "datasets"

# Archivos de entrada
FILE_TERENCICP_GAMES      = DATA_DATASETS_DIR / "games_terencicp.csv"
FILE_TERENCICP_CATEGORIES = DATA_DATASETS_DIR / "t-games-categories.csv"
FILE_TERENCICP_TAGS       = DATA_DATASETS_DIR / "t-games-tags.csv"
FILE_FRONKONGAMES         = DATA_DATASETS_DIR / "games_fronkon.csv"

# Se revisaron las distribuciones reales del dataset y se ajustaron
# los umbrales para reflejar mejor las categorías oficiales de Steam.
# El umbral de MIN_REVIEWS_VERY baja de 50 a 10 para no perder
# juegos de nicho con pocas pero consistentes reseñas positivas.

# Umbrales para la variable objetivo multiclase
MIN_REVIEWS          = 10     # Reseñas mínimas para considerar un juego válido
THRESH_OVERWHELM     = 0.95   # Overwhelmingly Positive
THRESH_VERY_POSITIVE = 0.80   # Very Positive
THRESH_MOSTLY        = 0.70   # Mostly Positive
THRESH_MIXED_LOW     = 0.40   # Límite inferior de Mixed

# Ajuste: MIN_REVIEWS_OVERWHELM sube de 500 a 1000 (más exigente)
# y MIN_REVIEWS_VERY baja de 50 a 10 (más inclusivo para juegos indie)
MIN_REVIEWS_OVERWHELM = 1000  # Reseñas mínimas para Overwhelmingly Positive (era 500)
MIN_REVIEWS_VERY      = 10    # Reseñas mínimas para Very Positive (era 50)
# === FIN COMMIT HOTFIX ===

# Etiquetas de las clases (en orden de menor a mayor éxito)
CLASS_LABELS = [
    "Negative",
    "Mixed",
    "Mostly Positive",
    "Very Positive",
    "Overwhelmingly Positive",
]

# Parámetros del modelo
TAG_FREQ_THRESHOLD   = 10    # Votos mínimos para conservar un tag
CUM_IMP_THRESHOLD    = 0.90  # Importancia acumulada para selección de features
TEST_SIZE            = 0.20
RANDOM_STATE         = 42
RF_N_ESTIMATORS      = 150
RF_MAX_DEPTH         = 10
RF_MIN_SAMPLES_SPLIT = 20
RF_MIN_SAMPLES_LEAF  = 10
