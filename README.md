# Dry Bean Dataset — Clasificación con Red Neuronal MLP

Clasificación automática de variedades de porotos a partir de características físicas y geométricas, usando una Red Neuronal Artificial Multicapa (MLP) implementada en TensorFlow/Keras.

Evaluación Parcial N°1  Deep Learning

## Descripción del problema

Una empresa de selección y comercialización de semillas necesita automatizar la clasificación de variedades de porotos, actualmente realizada de forma manual a partir de imágenes procesadas por visión computacional. Este proyecto entrena una red MLP que, a partir de 16 características numéricas (área, perímetro, ejes mayor/menor, excentricidad, factores de forma, etc.), predice a cuál de las 7 variedades pertenece cada observación:

`Seker`, `Barbunya`, `Bombay`, `Cali`, `Dermason`, `Horoz`, `Sira`

## Dataset

- **Fuente:** [Dry Bean Dataset — UCI Machine Learning Repository (ID 602)](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Autores:** M. Koklu, I. A. Özkan (2020). DOI: [10.24432/C50S4B](https://doi.org/10.24432/C50S4B)
- **Tamaño:** 13.611 observaciones, 16 variables numéricas, 7 clases, sin valores faltantes.

## Contenido del repositorio

```
├── EP1_DLY0100_DryBean_MLP.ipynb 
└── README.md
```

## Metodología

1. Carga y exploración del dataset (valores nulos, balance de clases, correlaciones).
2. Preprocesamiento: codificación de `Class`, split estratificado 70/15/15 (train/val/test) y estandarización con `StandardScaler`.
3. Arquitectura MLP base (2 capas ocultas, ReLU, salida Softmax).
4. Experimentos controlados variando un hiperparámetro a la vez: *learning rate*, *batch size* y número de épocas.
5. Comparación de funciones de activación (ReLU, tanh, LeakyReLU).
6. Evaluación de técnicas de regularización (Dropout, Batch Normalization).
7. Selección del modelo final con `EarlyStopping`, y evaluación en el conjunto de prueba.

## Resultados del modelo final

Configuración final: `learning_rate=0.001`, `batch_size=32`, activación **ReLU**, sin Dropout ni BatchNormalization (no aportaron mejoras frente al modelo base), entrenado con `EarlyStopping`.

| Métrica | Valor |
|---|---|
| Accuracy | 0.9261 |
| Precision (macro) | 0.9379 |
| Recall (macro) | 0.9381 |
| F1-score (macro) | 0.9379 |

La clase `Bombay` (minoritaria en el dataset) obtiene precision, recall y F1-score perfectos (1.00). El mayor margen de error del modelo se concentra entre `Sira` y `Dermason`, las dos variedades geométricamente más similares.

## Tecnologías utilizadas

- Python 3
- TensorFlow / Keras
- scikit-learn
- pandas, NumPy
- Matplotlib, Seaborn

## Integrantes
-Lucia Salazar Delgado
-Marcos Álvarez Muñoz
-Martina Moncada Vega

## Referencias

- Koklu, M. & Özkan, I. A. (2020). *Dry Bean Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C50S4B
