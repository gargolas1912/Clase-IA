# Pipeline Multiagente de Machine Learning — Telco Customer Churn

Notebook de Google Colab con un pipeline de clasificación basado en 3 agentes para predecir el abandono (churn) de clientes de telecomunicaciones.

**Dataset:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)


## Arquitectura

```
Dataset CSV
    │
    ▼
┌─────────────────────┐
│ AGENTE 1            │
│ Normalizador        │ → limpia, imputa, escala, codifica
└─────────────────────┘
    │ dataset limpio
    ▼
┌─────────────────────┐
│ AGENTE 2            │
│ Entrenador          │ → entrena, valida, selecciona modelo
└─────────────────────┘
    │ métricas + modelo
    ▼
┌─────────────────────┐
│ AGENTE 3            │
│ Comunicador         │ → genera reporte y responde preguntas
└─────────────────────┘
```

* **Agente 1 (Normalizador):** elimina el identificador de cliente, convierte `TotalCharges` a numérico, imputa valores faltantes, codifica variables categóricas con `LabelEncoder`, escala con `StandardScaler` y divide en train/test.
* **Agente 2 (Entrenador):** entrena Random Forest y Logistic Regression, evalúa con accuracy, precision, recall y F1, y selecciona el modelo con mejor F1.
* **Agente 3 (Comunicador):** usa `google/flan-t5-base` para generar un reporte académico en lenguaje natural y responder preguntas interactivas sobre el dataset y las métricas.


## Resultados

El Agente 2 entrenó 2 modelos y los comparó por F1:

|Modelo|Accuracy|Precision|Recall|F1|
|-|-|-|-|-|
|Random Forest|0.7899|0.6345|0.4920|0.5542|
|Logistic Regression|0.7991|0.6426|0.5481|0.5916|

El mejor modelo fue **Logistic Regression**, con el F1 más alto (0.5916). También tuvo mejor accuracy y precision que Random Forest, aunque la diferencia en recall fue la más significativa.


## Cambios y problemas que surgieron

* El pipeline `text2text-generation` no estaba disponible en la versión instalada de `transformers` (`KeyError: Unknown task`). Se solucionó reemplazando `pipeline(...)` por llamadas directas a `tokenizer()` + `modelo.generate()` con `torch.no\_grad()`.
* Al unificar el Agente 3 en una sola clase para reporte + chat, se actualizó la firma de `AgenteComunicador` para recibir `tokenizer` y `modelo` en vez de un objeto `generador`, lo que también requirió actualizar la instanciación en el pipeline completo.
* Las primeras respuestas del chat interactivo con FLAN-T5-base copiaban literalmente fragmentos completos del contexto (ej. el bloque de métricas) sin importar la pregunta hecha, devolviendo siempre la misma respuesta.
* Se solucionó parcialmente separando el contexto en dos bloques independientes (`contexto\_dataset` y `contexto\_metricas`) y seleccionando cuál inyectar en el prompt según palabras clave detectadas en la pregunta del usuario.
* Se ajustó `num\_beams=1` para las respuestas del chat (en vez de `num\_beams=4`, usado en el reporte automático), reduciendo el comportamiento extractivo de beam search en prompts más cortos y específicos.
* Hubo un `IndentationError` en el método `\_generar()`: el cuerpo de la función (`inputs = ...`, `with torch.no\_grad():`, `return ...`) no estaba indentado dentro de la definición del método.
* Se detectó un `FutureWarning` de pandas en `datos\["TotalCharges"].fillna(..., inplace=True)`, ya que ese patrón quedará deprecado en pandas 3.0. Pendiente de migrar a `datos\["TotalCharges"] = datos\["TotalCharges"].fillna(...)`.


## Cómo ejecutar

1. Abrir el notebook en Google Colab.
2. Ejecutar todas las celdas en orden (`Entorno de ejecución → Ejecutar todas`), ya que el Agente 3 depende de que `tokenizer` y `modelo` (FLAN-T5) estén cargados previamente.
3. El pipeline completo se dispara desde la función `pipeline\_completo(df)`, que orquesta los 3 agentes de forma secuencial.
4. Al finalizar el reporte automático, se abre un chat interactivo (`input()`) para hacer preguntas sobre el dataset y las métricas. Escribir `salir` para terminar.

