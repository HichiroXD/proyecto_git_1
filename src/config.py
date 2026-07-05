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

# Umbrales para la variable objetivo
MIN_REVIEWS          = 10    # Reseñas mínimas para considerar un juego válido
THRESH_OVERWHELM     = 0.95  # Overwhelmingly Positive
THRESH_POSITIVE      = 0.70  # Positive (fusiona Mostly + Very Positive)
THRESH_MIXED_LOW     = 0.40  # Límite inferior de Mixed

MIN_REVIEWS_OVERWHELM = 500  # Reseñas mínimas para Overwhelmingly Positive

# Número de clases del modelo (cambiar entre 3 y 4 para comparar)
N_CLASSES = 4

# Etiquetas para 4 clases (Negative / Mixed / Positive / Overwhelmingly Positive)
CLASS_LABELS_4 = [
    "Negative",
    "Mixed",
    "Positive",
    "Overwhelmingly Positive",
]

# Etiquetas para 3 clases (Negative / Mixed / Positive)
CLASS_LABELS_3 = [
    "Negative",
    "Mixed",
    "Positive",
]

# Etiquetas activas según N_CLASSES
CLASS_LABELS = CLASS_LABELS_4 if N_CLASSES == 4 else CLASS_LABELS_3

# Parámetros del modelo
TAG_FREQ_THRESHOLD   = 10    # Votos mínimos para conservar un tag
CUM_IMP_THRESHOLD    = 0.90  # Importancia acumulada para selección de features
TEST_SIZE            = 0.20
RANDOM_STATE         = 42
RF_N_ESTIMATORS      = 200
RF_MAX_DEPTH         = 10
RF_MIN_SAMPLES_SPLIT = 20
RF_MIN_SAMPLES_LEAF  = 10
