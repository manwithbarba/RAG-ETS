# Protocolo de Evaluación del Sistema RAG de ETS

## 1. Introducción

Este documento describe el protocolo para evaluar el rendimiento del sistema de Generación Aumentada por Recuperación (RAG). El objetivo es medir la calidad y fiabilidad de las respuestas del sistema de una manera estructurada y repetible.

La evaluación se divide en dos enfoques complementarios:

1.  **Evaluación Cualitativa:** Basada en el feedback directo del usuario experto.
2.  **Evaluación Cuantitativa:** Basada en métricas automatizadas utilizando un framework especializado (`ragas`).

## 2. Requisitos Previos

1.  **Proyecto Configurado:** El proyecto debe estar completamente instalado (`pip install -r requirements.txt`).
2.  **Base de Datos Creada:** El script `ingest.py` debe haber sido ejecutado con los documentos finales.
3.  **API Key de OpenAI:** Para la evaluación cuantitativa, el framework `ragas` utiliza un LLM (por defecto, de OpenAI) para "juzgar" la calidad de las respuestas. Es necesario tener una API key de OpenAI.

---

## 3. Evaluación Cualitativa (Feedback de Usuario)

Este método proporciona información rápida y directa sobre el rendimiento percibido del sistema.

### Proceso

1.  **Iniciar la Aplicación:**
    ```bash
    streamlit run app.py
    ```
2.  **Realizar Consultas:** Interactúa con la aplicación realizando una amplia variedad de preguntas. Prueba preguntas fáciles, difíciles, específicas y generales.
3.  **Registrar Feedback:** Después de cada respuesta generada, utiliza los botones de calificación:
    - **👍 (Buena respuesta):** Si la respuesta es correcta, relevante y está bien fundamentada en las fuentes.
    - **👎 (Mala respuesta):** Si la respuesta es incorrecta, irrelevante, incompleta o no cita correctamente las fuentes.

### Resultado

Cada vez que se presiona un botón, se añade una nueva línea al archivo `feedback.csv` en la raíz del proyecto. Este archivo contiene:

- `timestamp`: Fecha y hora de la evaluación.
- `question`: La pregunta realizada.
- `answer`: La respuesta generada por el sistema.
- `context`: Las fuentes que el sistema utilizó.
- `rating`: Tu calificación (👍 o 👎).

El análisis de este archivo permite identificar patrones en las preguntas que el sistema maneja bien o mal.

---

## 4. Evaluación Cuantitativa (Métricas con `ragas`)

Este método proporciona puntuaciones numéricas y objetivas sobre diferentes aspectos del rendimiento del sistema.

### Paso 1: Crear el "Golden Set" de Evaluación

Esta es la tarea más importante. Consiste en crear un dataset de referencia.

- **Acción:** Crea un archivo llamado `eval_dataset.csv` en la raíz del proyecto.
- **Formato:** El archivo debe tener exactamente las siguientes 3 columnas:
  - `question`: Una pregunta de prueba.
  - `ground_truth_answer`: La respuesta "ideal" que un experto humano escribiría.
  - `ground_truth_context`: El texto exacto de los documentos que justifica la respuesta ideal.

- **Ejemplo de una fila en `eval_dataset.csv`:**
  ```csv
  question,ground_truth_answer,ground_truth_context
  "¿Cuál es la recomendación para el uso de fármaco X en monoterapia?","El fármaco X no se recomienda en monoterapia para pacientes con la condición Y, según la guía de práctica clínica.","La guía de práctica clínica establece en la sección 4.1 que 'el uso de fármaco X en monoterapia no está recomendado para la condición Y'."
  ```

### Paso 2: Configurar el Entorno

- **Acción:** Abre una terminal en la carpeta del proyecto y configura tu clave de API de OpenAI como una variable de entorno.
  ```bash
  # En Windows
  set OPENAI_API_KEY=sk-...

  # En macOS/Linux
  export OPENAI_API_KEY=sk-...
  ```

### Paso 3: Ejecutar el Script de Evaluación

- **Acción:** Una vez que tu `eval_dataset.csv` esté listo y la clave API configurada, ejecuta el script:
  ```bash
  python evaluate.py
  ```

### Paso 4: Interpretar los Resultados

El script generará un informe con una puntuación de 0 a 1 para cada una de las siguientes métricas:

- **`faithfulness` (Fidelidad):** Mide si la respuesta generada se apega a los hechos presentados en el contexto recuperado. Una puntuación baja indica que el LLM está "alucinando" o inventando información.
- **`answer_relevancy` (Relevancia de la Respuesta):** Mide qué tan relevante es la respuesta para la pregunta. Una puntuación baja indica que la respuesta se desvía del tema.
- **`context_precision` (Precisión del Contexto):** Mide si los fragmentos de texto recuperados son relevantes para la pregunta. Es una medida de la calidad del sistema de búsqueda (retriever).
- **`context_recall` (Recuperación del Contexto):** Mide si el retriever fue capaz de encontrar TODA la información necesaria para componer la respuesta ideal. Una puntuación baja indica que el retriever no está encontrando los fragmentos correctos.

**Acciones según los resultados:**
- Si `faithfulness` es bajo -> Refinar el prompt en `app.py` para ser más estricto.
- Si `context_recall` o `context_precision` son bajos -> Experimentar con la estrategia de `chunking` en `ingest.py` o con el valor de `k` en la interfaz.
