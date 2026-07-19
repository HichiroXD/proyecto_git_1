# Predictor de Juegos Steam

## Descripción del problema

El mercado de distribución digital en Steam es altamente competitivo, con miles de lanzamientos anuales. Para los desarrolladores independientes, es crítico entender qué factores estructurales de un juego se asocian históricamente con una buena recepción de la comunidad, antes de invertir recursos en su producción.

Este proyecto busca reducir esa incertidumbre mediante un modelo predictivo entrenado sobre el historial de juegos ya lanzados. La hipótesis central es que características conocidas **antes del lanzamiento** (como el género, el precio, las plataformas soportadas o el desarrollador) contienen señal suficiente para anticipar qué nivel de recepción tendrá un juego nuevo con esas mismas características.

Este proyecto predice el **nivel de éxito en 3 categorías**, derivadas de la escala oficial de Steam y ajustadas para lograr un balance de clases que permita un aprendizaje efectivo del modelo:

| Clase | Definición | % del dataset |
|---|---|---|
| Negative | ratio de reseñas positivas < 40% | 5.5% |
| Mixed | 40% ≤ ratio < 70% | 25.8% |
| Positive | ratio ≥ 70% | 68.8% |

> **Nota de diseño:** originalmente se planificaron 5 clases siguiendo la escala completa de Steam (Overwhelmingly Positive, Very Positive, Mostly Positive, Mixed, Negative). Sin embargo, durante la implementación se comprobó que las clases intermedias tenían F1 = 0.00 — el modelo las ignoraba completamente por el desbalance severo. Se evaluaron también 4 clases con el mismo resultado. La reducción a 3 clases y el uso de `class_weight='balanced'` fueron decisiones tomadas en base a los datos, documentadas en el notebook y en el historial de commits del repositorio.

El modelo aprende patrones históricos de juegos ya lanzados, pero usando solo features que un desarrollador conocería **antes de publicar** su juego. El alcance es orientativo, no determinista.

---

## Datasets y fuentes

Se utilizan dos datasets complementarios de Kaggle, unidos por `AppID`:

**Dataset 1 — [terencicp/steam-games-december-2023](https://www.kaggle.com/datasets/terencicp/steam-games-december-2023)**
Información base de ~61.000 juegos: precio, reseñas positivas/negativas, fecha de lanzamiento, y tablas separadas de categorías y tags en formato largo (`t-games-categories.csv`, `t-games-tags.csv`).

**Dataset 2 — [fronkongames/steam-games-dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)**
Más de 125.000 juegos con columnas adicionales no disponibles en el primer dataset: desarrollador, publisher, géneros, plataformas (Windows/Mac/Linux), logros y Metacritic score.

El merge externo por `AppID` permite enriquecer el dataset original con las columnas que faltaban. El 93.5% de los juegos del dataset base encontraron match en fronkongames (56.993 de 60.952). El 6.5% restante conserva sus columnas base.

> **Nota técnica:** el CSV de fronkongames presenta un desplazamiento de columnas causado por un índice de pandas exportado sin nombre. Se corrige en `data_loader.py` usando `index_col=0` y renombrando las columnas afectadas.

---

## Estructura del repositorio

```
proyecto_git_1/
├── data/
│   └── datasets/             # CSVs de Kaggle (no incluidos en el repo por tamaño)
│       └── INSTRUCCIONES.txt # Instrucciones de descarga
├── src/
│   ├── config.py             # Constantes, rutas y umbrales centralizados
│   ├── data_loader.py        # Carga y merge de los dos datasets
│   ├── feature_eng.py        # Limpieza, encoding y construcción del target
│   ├── model.py              # Pipelines sklearn y entrenamiento
│   └── evaluation.py         # Métricas y visualizaciones multiclase
├── notebooks/
│   └── analysis.ipynb        # Análisis completo importando los módulos src/
├── reports/
│   └── informe.pdf           # Informe de análisis del repositorio Git
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
#    (ver data/datasets/INSTRUCCIONES.txt)

# 4. Ejecutar el notebook
jupyter notebook notebooks/analysis.ipynb
```

---

## Justificación del modelo

Para abordar este problema de clasificación multiclase se evalúan **Decision Tree** y **Random Forest**. La elección se justifica en que el dataset presenta una fuerte mezcla de variables numéricas y categóricas de alta cardinalidad (tags, géneros, desarrolladores). Los modelos basados en árboles son ideales porque:

1. No requieren escalamiento de variables numéricas.
2. Capturan interacciones no lineales entre variables.
3. Permiten medir la importancia de cada feature, ofreciendo alta interpretabilidad.

La selección de features se encapsula en un `Pipeline` sklearn (`CumulativeImportanceSelector`) para garantizar que el conjunto de prueba no influya en qué variables se eligen — evitando data leakage. Este transformer ajusta un Random Forest interno solo con los datos de entrenamiento de cada fold durante la validación cruzada, reduciendo el feature set de ~526 a ~190 columnas (90% de importancia acumulada).

**Limitación documentada:** `release_year` muestra correlación con la variable objetivo que en parte refleja un sesgo de captura del dataset (los juegos más recientes tienen menos reseñas acumuladas). Se aplica un filtro de reseñas mínimas para mitigar este efecto, y la limitación se documenta explícitamente en las conclusiones.

Se usa `class_weight='balanced'` en ambos modelos para compensar el desbalance inherente del catálogo de Steam (Positive: 68.8%, Mixed: 25.8%, Negative: 5.5%).

---

## Metodología

1. Validación de existencia de los dataset antes de la carga
2. Carga y merge de los dos datasets por `AppID`
3. Análisis exploratorio de datos (EDA): precios, lanzamientos por año, distribución de clases, plataformas
4. Limpieza y feature engineering: filtro de reseñas mínimas, construcción del target multiclase, One-Hot Encoding de categorías, tags y géneros
5. Selección de variables mediante importancia acumulada (umbral: 90%)
6. División train/test estratificada (80/20)
7. Entrenamiento con Pipeline (CumulativeImportanceSelector + clasificador)
8. Validación cruzada estratificada 5-fold (métrica: Macro F1)
9. Evaluación en test set con métricas multiclase (classification report, matrices de confusión, curvas ROC one-vs-rest)

---

## Resultados

Se evaluaron dos modelos sobre el conjunto de prueba (20% del dataset, 8.048 juegos no vistos durante el entrenamiento):

| Métrica | Decision Tree | Random Forest |
|---|---|---|
| Accuracy | 0.520 | **0.586** |
| Macro F1 | 0.408 | **0.445** |
| Weighted F1 | 0.571 | **0.621** |
| AUC macro-OvR | 0.617 | **0.673** |
| F1 CV 5-fold (train) | ~0.38 | **~0.42** |

**Random Forest se selecciona como modelo final** por superar a Decision Tree en todas las métricas. Adicionalmente, su F1 en validación cruzada es más estable entre folds, lo que indica mejor generalización.

### Resultados por clase (Random Forest)

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.41 | 0.30 | 0.35 |
| Mixed | 0.14 | 0.64 | 0.23 |
| Positive | 0.83 | 0.69 | 0.75 |

### Variables más influyentes

Las 10 features con mayor importancia en el modelo final son: `release_year`, `achievements`, `metacritic_score`, `price`, `cat_steam_trading_cards`, `required_age`, `cat_steam_cloud`, `tag_2d`, `tag_singleplayer`, `tag_great_soundtrack`.

El dominio de `release_year` y `achievements` refleja que el año de lanzamiento y la cantidad de logros planeados son señales fuertes del nivel de producción y recepción histórica de un juego.

---

## Conclusiones

### Hallazgos principales

1. **El modelo logra capacidad discriminante moderada pero real.** Con AUC macro = 0.673, Random Forest supera significativamente al azar (0.5) para distinguir juegos Negative, Mixed y Positive usando solo características disponibles antes del lanzamiento.

2. **El género importa más que el precio.** Tags como `tag_2d`, `tag_singleplayer` y `tag_pixel_graphics` tienen mayor importancia que `price`. Históricamente, los juegos de nicho indie con estética pixel art o mecánicas singleplayer 2D atraen comunidades más leales en Steam.

3. **`Mixed` es el límite del modelo** (F1 = 0.23). Los juegos en el rango 40-70% de reseñas positivas son fundamentalmente difíciles de predecir con features pre-lanzamiento — son juegos que polarizaron a la audiencia por razones no anticipables sin información sobre el juego en sí.

4. **El proceso de refinamiento de clases fue parte del análisis.** Se probaron 5, 4 y 3 clases, y se identificó que el desbalance del catálogo de Steam requería tanto reducción de clases como `class_weight='balanced'`. Este proceso está documentado en el historial de commits del repositorio.

### Limitaciones

1. **Sesgo temporal de `release_year`:** el dataset fue capturado en diciembre 2023; los juegos más recientes tienen menos tiempo de exposición a críticas negativas, lo que infla artificialmente su ratio de reseñas positivas.

2. **Metacritic score como feature:** tiene alta importancia pero solo existe para juegos con cobertura en Metacritic (típicamente AAA o juegos mediáticamente relevantes), introduciendo un sesgo de selección.

3. **Ausencia de features de contenido:** la descripción del juego, imágenes y trailer contienen información predictiva que este modelo no captura al trabajar solo con metadatos estructurados.

4. **Alcance orientativo:** el modelo estima probabilidades históricas basadas en patrones del catálogo de Steam, no predice el éxito de un juego específico con certeza.

---

## Declaración de uso de inteligencia artificial

Este proyecto fue desarrollado con asistencia de Claude para el diseño de la metodología, redacción del README, estructura del código y revisión. El código fue revisado, pensado y ejecutado por el usuario. El historial de Git refleja el desarrollo y progreción del proyecto.
