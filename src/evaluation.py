"""
evaluation.py
-------------
Métricas y visualizaciones para evaluar modelos de clasificación multiclase.

Incluye:
  - print_report(): classification report completo por clase.
  - plot_confusion_matrix(): matriz de confusión normalizada.
  - plot_roc_curves(): curvas ROC one-vs-rest por clase.
  - plot_feature_importance(): top N features del modelo final.
  - compare_models(): tabla comparativa de métricas entre modelos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from src.config import CLASS_LABELS


def print_report(pipeline: Pipeline, X_test, y_test) -> None:
    """
    Imprime el classification report completo con métricas por clase.

    Reporta precision, recall y F1 para cada una de las 5 clases,
    más los promedios macro y weighted. Se usa macro-F1 como métrica
    principal por ser más robusta ante el desbalance de clases.

    Parameters
    ----------
    pipeline : Pipeline sklearn ya entrenado
    X_test   : pd.DataFrame — features del conjunto de prueba
    y_test   : pd.Series    — target real del conjunto de prueba
    """
    y_pred = pipeline.predict(X_test)
    print(classification_report(
        y_test, y_pred,
        target_names=CLASS_LABELS,
        zero_division=0,
    ))


def plot_confusion_matrix(pipeline: Pipeline, X_test, y_test) -> None:
    """
    Genera y muestra la matriz de confusión normalizada por fila.

    La normalización por fila (normalize='true') muestra la proporción
    de predicciones correctas e incorrectas para cada clase real,
    lo que facilita identificar qué clases el modelo confunde más.

    Parameters
    ----------
    pipeline : Pipeline sklearn ya entrenado
    X_test   : pd.DataFrame — features del conjunto de prueba
    y_test   : pd.Series    — target real del conjunto de prueba
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test,
        display_labels=CLASS_LABELS,
        normalize="true",
        ax=ax,
        colorbar=True,
        cmap="Blues",
    )
    ax.set_title("Matriz de confusión normalizada")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def plot_roc_curves(pipeline: Pipeline, X_test, y_test) -> None:
    """
    Curvas ROC one-vs-rest (OvR) para cada una de las 5 clases.

    En clasificación multiclase no existe una única curva ROC — se genera
    una por clase usando la estrategia One-vs-Rest: para cada clase, se
    considera esa clase como positiva y el resto como negativas.
    Se reporta el AUC macro-promedio como métrica de resumen.

    Parameters
    ----------
    pipeline : Pipeline sklearn ya entrenado (debe tener predict_proba)
    X_test   : pd.DataFrame — features del conjunto de prueba
    y_test   : pd.Series    — target real del conjunto de prueba
    """
    y_prob = pipeline.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=CLASS_LABELS)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["steelblue", "coral", "seagreen", "gold", "mediumpurple"]

    auc_scores = []
    for i, (label, color) in enumerate(zip(CLASS_LABELS, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)
        ax.plot(fpr, tpr, color=color, lw=1.5,
                label=f"{label} (AUC = {auc_score:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Azar")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curvas ROC one-vs-rest por clase")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()

    macro_auc = np.mean(auc_scores)
    print(f"AUC macro-promedio: {macro_auc:.3f}")


def plot_feature_importance(pipeline: Pipeline, top_n: int = 30) -> None:
    """Gráfico de barras con las top_n features del modelo final."""
    raise NotImplementedError("Pendiente")


def compare_models(results: dict) -> pd.DataFrame:
    """
    Recibe un dict {nombre_modelo: métricas} y devuelve una tabla comparativa.
    Ejemplo de entrada:
        {
            "Decision Tree": {"accuracy": 0.61, "macro_f1": 0.58, "auc_ovr": 0.79},
            "Random Forest": {"accuracy": 0.65, "macro_f1": 0.62, "auc_ovr": 0.83},
        }
    """
    raise NotImplementedError("Pendiente")
