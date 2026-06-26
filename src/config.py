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
DATA_RAW_DIR = ROOT_DIR / "data" / "datasets"

# Archivos de entrada
FILE_TERENCICP_GAMES      = DATA_RAW_DIR / "games_terencicp.csv"
FILE_TERENCICP_CATEGORIES = DATA_RAW_DIR / "t-games-categories.csv"
FILE_TERENCICP_TAGS       = DATA_RAW_DIR / "t-games-tags.csv"
FILE_FRONKONGAMES         = DATA_RAW_DIR / "games_fronkon.csv"

# Umbrales para la variable objetivo multiclase
# Se definen aquí para que cualquier modificación sea centralizada
MIN_REVIEWS          = 10     # Reseñas mínimas para considerar un juego válido
THRESH_OVERWHELM     = 0.95   # Overwhelmingly Positive
THRESH_VERY_POSITIVE = 0.80   # Very Positive
THRESH_MOSTLY        = 0.70   # Mostly Positive
THRESH_MIXED_LOW     = 0.40   # Límite inferior de Mixed

MIN_REVIEWS_OVERWHELM = 500   # Reseñas mínimas para Overwhelmingly Positive
MIN_REVIEWS_VERY      = 50    # Reseñas mínimas para Very Positive

# Etiquetas de las clases (en orden de menor a mayor éxito)
CLASS_LABELS = [
    "Negative",
    "Mixed",
    "Mostly Positive",
    "Very Positive",
    "Overwhelmingly Positive",
]

# Parámetros del modelo
TAG_FREQ_THRESHOLD  = 10    # Votos mínimos para conservar un tag
CUM_IMP_THRESHOLD   = 0.90  # Importancia acumulada para selección de features
TEST_SIZE           = 0.20
RANDOM_STATE        = 42
RF_N_ESTIMATORS     = 200
RF_MAX_DEPTH        = 10
RF_MIN_SAMPLES_SPLIT = 20
RF_MIN_SAMPLES_LEAF  = 10
