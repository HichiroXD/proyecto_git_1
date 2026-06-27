# Predictor de Juegos Steam

## Descripción del problema

El mercado de distribución digital en Steam es altamente competitivo, con miles de lanzamientos anuales. Para los desarrolladores independientes, es crítico entender qué factores estructurales de un juego se asocian históricamente con una buena recepción de la comunidad, antes de invertir recursos en su producción.

Este proyecto busca reducir esa incertidumbre mediante un modelo predictivo entrenado sobre el historial de juegos ya lanzados. La hipótesis central es que características conocidas **antes del lanzamiento** (como el género, el precio, las plataformas soportadas o el desarrollador) contienen señal suficiente para anticipar qué nivel de recepción tendrá un juego nuevo con esas mismas características.

A diferencia de una clasificación binaria (exitoso/no exitoso), este proyecto predice el **nivel de éxito en 5 categorías**, siguiendo la escala oficial de Steam:

| Clase | Definición |
|---|---|
| Overwhelmingly Positive | ≥ 95% positivas y ≥ 500 reseñas |
| Very Positive | ≥ 80% positivas y ≥ 50 reseñas |
| Mostly Positive | ≥ 70% positivas |
| Mixed | entre 40% y 69% positivas |
| Negative | < 40% positivas |

El modelo aprende patrones históricos de juegos ya lanzados, pero usando solo features que un desarrollador conocería **antes de publicar** su juego. El alcance es orientativo, no determinista.

---

## Datasets y fuentes

Se utilizan dos datasets complementarios de Kaggle, unidos por `AppID`:

**Dataset 1 — [terencicp/steam-games-december-2023](https://www.kaggle.com/datasets/terencicp/steam-games-december-2023)**
Información base de ~61.000 juegos: precio, reseñas positivas/negativas, fecha de lanzamiento, y tablas separadas de categorías y tags.

**Dataset 2 — [fronkongames/steam-games-dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)**
Más de 110.000 juegos con columnas adicionales: desarrollador, publisher, géneros, plataformas (Windows/Mac/Linux), idiomas soportados, logros y Metacritic score.

El merge externo por `AppID` permite enriquecer el dataset original con las columnas que faltaban, ampliando el feature set disponible para el modelo.

---

## Estructura del repositorio

```
mi_proyecto_git/
├── data/
│   └── datasets/             # CSVs de Kaggle (no incluidos en el repo)
├── src/
│   ├── config.py             # Constantes, rutas y umbrales centralizados
│   ├── data_loader.py        # Carga y merge de los dos datasets
│   ├── feature_eng.py        # Limpieza, encoding y construcción del target
│   ├── model.py              # Pipelines sklearn y entrenamiento
│   └── evaluation.py         # Métricas y visualizaciones
├── notebooks/
│   └── analysis.ipynb        # Análisis completo importando los módulos src/
├── reports/
│   └── informe.pdf           # Informe de análisis del repositorio (Basado en los resultados del README)
├── .gitignore
├── environment.yml
└── README.md
```

---

## Instrucciones de instalación y ejecución

```bash
# 1. Clonar el repositorio
git clone https://github.com/<usuario>/proyecto_git_1.git
cd proyecto_git_1

# 2. Crear el entorno virtual
conda env create -f environment.yml
conda activate proyecto_git_1

# 3. Descargar los datasets desde Kaggle y colocarlos en data/datasets/
#    (ver data/raw/INSTRUCCIONES.txt)

# 4. Ejecutar el notebook
jupyter notebook notebooks/analysis.ipynb
```

---

## Justificación del modelo

Para abordar este problema de clasificación multiclase se evalúan **Decision Tree** y **Random Forest**. La elección se justifica en que el dataset presenta una fuerte mezcla de variables numéricas y categóricas de alta cardinalidad (tags, géneros, desarrolladores). Los modelos basados en árboles son ideales porque:

1. No requieren escalamiento de variables numéricas.
2. Capturan interacciones no lineales entre variables.
3. Permiten medir la importancia de cada feature, ofreciendo alta interpretabilidad.

La selección de features se encapsula en un `Pipeline` sklearn (`CumulativeImportanceSelector`) para garantizar que el conjunto de prueba no influya en qué variables se eligen — evitando data leakage.

**Limitación documentada:** `release_year` muestra correlación con la variable objetivo que en parte refleja un sesgo de captura del dataset (los juegos más recientes tienen menos reseñas acumuladas). Se aplica un filtro de reseñas mínimas para mitigar este efecto, y la limitación se documenta explícitamente en las conclusiones.

---

## Metodología

1. Carga y merge de los dos datasets por `AppID`
2. Análisis exploratorio de datos (EDA)
3. Limpieza y feature engineering
4. Selección de variables (importancia acumulada, 90%)
5. División train/test estratificada (80/20)
6. Entrenamiento con Pipeline (selector + clasificador)
7. Validación cruzada estratificada 5-fold
8. Evaluación en test set con métricas multiclase

---

## Resultados

*(Por completar tras la implementación)*

---

## Conclusiones

*(Por completar tras la implementación)*

---

## Declaración de uso de inteligencia artificial

Este proyecto fue desarrollado con asistencia de Claude para el diseño de la metodología, redacción del README, estructura del código y revisión. El código fue revisado, pensado y ejecutado por el usuario. El historial de Git refleja el desarrollo y progreción del proyecto.
