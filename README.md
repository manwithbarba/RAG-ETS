# Sistema RAG para Consulta de Evaluaciones de Tecnologías Sanitarias (ETS)

**Autor:** Julián Sánchez Viamonte

## 1. Objetivos del Proyecto

Este proyecto implementa un sistema de **Generación Aumentada por Recuperación (RAG)** diseñado para permitir la consulta en lenguaje natural de una base de conocimiento privada compuesta por informes y documentos de Evaluaciones de Tecnologías Sanitarias (ETS).

Los objetivos principales son:

- **Respuestas Basadas en Evidencia:** Asegurar que todas las respuestas generadas por el sistema se basen estrictamente en el contenido de los documentos proporcionados, evitando la invención de información (alucinaciones).
- **Trazabilidad y Confianza:** Proveer citas en línea en las respuestas generadas, permitiendo al usuario final verificar qué fragmento de qué documento fuente respalda cada afirmación.
- **Interfaz Intuitiva:** Ofrecer una interfaz de usuario web simple e interactiva para que usuarios no técnicos puedan realizar consultas complejas fácilmente.
- **Arquitectura Local y Abierta:** Construir el sistema utilizando exclusivamente modelos y herramientas de código abierto, garantizando la privacidad de los datos y la soberanía sobre la infraestructura.

## 2. Stack Tecnológico

- **Lenguaje:** Python
- **Framework RAG:** LangChain
- **Interfaz Web:** Streamlit
- **Base de Datos Vectorial:** ChromaDB
- **Modelo de Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (para convertir texto en vectores)
- **Modelo de Lenguaje (LLM):** `Meta-Llama-3.1-8B-Instruct-GGUF` (cargado localmente vía `llama-cpp-python`)

## 3. Estructura del Proyecto

```
rag-evaluaciones-salud/
│
├── documentos/               # Carpeta para depositar los archivos fuente (.pdf, .docx).
├── models/                   # (Opcional) Se puede usar para guardar modelos descargados.
├── vectorstores/
│   └── db/                   # Base de datos vectorial ChromaDB generada por la ingesta.
│
├── app.py                    # Script principal de la aplicación web Streamlit.
├── ingest.py                 # Script para procesar los documentos y crear la base de datos.
├── requirements.txt          # Lista de dependencias de Python.
└── README.md                 # Este archivo.
```

- **`app.py`**: Lanza la interfaz web. Carga el LLM y la base de datos vectorial, procesa la entrada del usuario, ejecuta la cadena RAG y muestra la respuesta con citas.
- **`ingest.py`**: Lee los documentos de la carpeta `documentos/`, los divide en fragmentos (chunks), genera los embeddings y los almacena en la base de datos ChromaDB en `vectorstores/db/`.

## 4. Instalación y Uso

Sigue estos pasos para ejecutar el proyecto en tu máquina local.

**Prerrequisitos:**
- Python 3.9+ instalado.
- Git instalado.

**Pasos:**

1.  **Clonar el Repositorio:**
    ```bash
    git clone <URL-del-repositorio-en-github>
    cd rag-evaluaciones-salud
    ```

2.  **Instalar Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Descargar el Modelo LLM:**
    Este proyecto está configurado para usar `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`. Debes descargarlo y asegurarte de que la ruta en `app.py` (variable `MODEL_PATH`) apunte a la ubicación correcta del archivo `.gguf` en tu sistema.

4.  **Añadir Documentos Fuente:**
    Coloca todos tus archivos `.pdf` y `.docx` en la carpeta `documentos/`.

5.  **Crear la Base de Datos Vectorial:**
    Ejecuta el script de ingesta una sola vez (o cada vez que cambies los documentos).
    ```bash
    python ingest.py
    ```

6.  **Lanzar la Aplicación:**
    ```bash
    streamlit run app.py
    ```
    Esto abrirá una nueva pestaña en tu navegador web con la interfaz de la aplicación.

## 5. Resultados y Funcionalidades Implementadas

- **Consulta en Lenguaje Natural:** Interfaz web para realizar preguntas sobre la base documental.
- **Generación Aumentada por Recuperación (RAG):** El sistema recupera los fragmentos más relevantes de los documentos antes de generar una respuesta, asegurando que esté fundamentada en el contexto.
- **Citas en Línea:** Las respuestas incluyen marcadores de cita (ej. `[1]`, `[2]`) que se corresponden con las fuentes mostradas, permitiendo la verificación de la información.
- **Control de Recuperación (k):** La interfaz incluye un control deslizable para ajustar dinámicamente cuántos fragmentos de texto se recuperan, permitiendo experimentar con el balance entre contexto y "ruido".

## 6. Próximos Pasos y Hoja de Ruta para Evaluación

La siguiente fase crucial es la **Evaluación (Fase 3)** para medir objetivamente la calidad y fiabilidad del sistema.

1.  **Crear un Set de Evaluación (Golden Set):**
    - Elaborar una lista de 20-30 preguntas representativas que cubran diferentes aspectos de los documentos.
    - Para cada pregunta, redactar la "respuesta ideal" de forma manual, basada en la lectura de los documentos.
    - Anotar qué documentos o fragmentos contienen la información necesaria para cada respuesta ideal.

2.  **Definir Métricas de Calidad RAG:**
    - **Faithfulness (Fidelidad):** ¿La respuesta generada se contradice con el contexto proporcionado? Se mide pidiéndole a un LLM (o a un humano) que verifique si cada afirmación de la respuesta está respaldada por el contexto.
    - **Answer Relevancy (Relevancia de la Respuesta):** ¿Qué tan relevante es la respuesta para la pregunta del usuario? No debe desviarse del tema.
    - **Context Recall (Recuperación del Contexto):** ¿El sistema recuperó todos los fragmentos de texto relevantes para responder a la pregunta? Se compara con las anotaciones del Golden Set.

3.  **Implementar un Bucle de Feedback de Usuario:**
    - Añadir botones de "pulgar arriba / pulgar abajo" (👍/👎) a cada respuesta en la interfaz de Streamlit.
    - Almacenar este feedback (pregunta, respuesta, contexto, calificación) en un archivo o base de datos simple. Este feedback es extremadamente valioso para identificar debilidades y guiar futuras optimizaciones.