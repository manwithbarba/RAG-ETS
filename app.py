import streamlit as st
import os
import pandas as pd
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import LlamaCpp

# --- Configuración y Carga ---

# Constantes
DB_PATH = "vectorstores/db/"
# Ruta al modelo GGUF local. Se asume que está en la carpeta del otro proyecto.
MODEL_PATH = "C:/Users/jsanc/Proyectos IA/Proyecto_MAPEAR_PRO/backend/models/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/snapshots/bf5b95e96dac0462e2a09145ec66cae9a3f12067/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Template del prompt para citas en línea
PROMPT_TEMPLATE = """### INSTRUCCIONES:
Tu tarea es actuar como un asistente experto en análisis de evidencia científica y responder la pregunta del usuario.
Te basarás ÚNICA Y EXCLUSIVAMENTE en el contexto que te proporciono, el cual consiste en varias fuentes numeradas.

Reglas estrictas:
1.  Sintetiza la información de las fuentes para construir una respuesta coherente y fluida.
2.  **CRÍTICO**: Después de cada oración o afirmación que extraigas de una fuente, DEBES añadir la cita correspondiente. Por ejemplo: "La eficacia del tratamiento fue del 80% [1]."
3.  Si una misma oración combina información de múltiples fuentes, cita todas las relevantes. Por ejemplo: "El estudio incluyó pacientes de Argentina [1] y Chile [2]."
4.  NO inventes información. Si la respuesta no se encuentra en las fuentes proporcionadas, responde exactamente con: "La información solicitada no se encuentra en los documentos disponibles."
5.  No incluyas en tu respuesta los nombres de los documentos, solo los números de cita.

### CONTEXTO:
{context}

### PREGUNTA:
{question}

### RESPUESTA CUMPLIENDO LAS REGLAS:
"""

@st.cache_resource
def load_resources():
    """Carga y cachea los recursos pesados (modelos y BD) para evitar recargarlos.
    
    Returns:
        Tuple: Una tupla con el objeto de la base de datos, el objeto del LLM y un 
               mensaje de error (si lo hay).
    """
    # Validar que el archivo del modelo exista antes de intentar cargarlo.
    if not os.path.exists(MODEL_PATH):
        return None, None, f"Error: No se encuentra el archivo del modelo en la ruta: {MODEL_PATH}"

    # Cargar el modelo de embeddings para la búsqueda de similitud.
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    # Cargar la base de datos vectorial persistida en disco.
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # Cargar el LLM local usando LlamaCpp. 
    # n_gpu_layers=-1 intenta descargar todas las capas a la GPU si está disponible.
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=4096,       # Longitud del contexto
        n_gpu_layers=-1,  # Aceleración por GPU
        verbose=False,
        temperature=0.1
    )
    
    return db, llm, None

def main():
    st.title("🤖 Consulta de Evidencia en Tecnologías Sanitarias (Local)")
    st.caption("Este prototipo usa el modelo Llama-3.1-8B cargado localmente.")
    st.caption("Autor: Julián Sánchez Viamonte")

    # Cargar los recursos al iniciar la app.
    db, llm, error_message = load_resources()

    if error_message:
        st.error(error_message)
        st.info("Asegúrate de que la ruta al modelo .gguf es correcta en el script app.py.")
        return

    # --- Interfaz de Usuario ---
    query_text = st.text_input("Escribe tu pregunta sobre los documentos:", placeholder="Ej: ¿Cuál es la eficacia de la IA en la supervisión clínica?")

    k_value = st.slider(
        label="Número de fragmentos a recuperar (k):",
        min_value=1,
        max_value=10,
        value=3,
        help="Controla cuántos fragmentos de texto se usan como contexto para la respuesta. Un número más alto puede dar más información pero también más 'ruido'."
    )

    # --- Lógica de la Aplicación ---

    # Inicializar session_state para guardar la última respuesta y evitar feedback duplicado
    if 'last_response' not in st.session_state:
        st.session_state.last_response = ""
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = False

    if st.button("🔍 Buscar") and query_text:
        st.session_state.feedback_given = False # Resetear feedback en cada nueva búsqueda
        with st.spinner("Buscando en los documentos y generando respuesta con el modelo local..."):
            
            # --- Lógica RAG (Retrieval-Augmented Generation) ---
            retriever = db.as_retriever(search_kwargs={'k': k_value})
            retrieved_docs = retriever.invoke(query_text)

            formatted_context = ""
            for i, doc in enumerate(retrieved_docs):
                formatted_context += f"### Fuente [{i+1}]\n"
                formatted_context += f"Documento: {doc.metadata.get('source', 'N/A').replace('documentos\\', '')}\n"
                formatted_context += f"Contenido: {doc.page_content}\n\n"

            prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
            
            rag_chain = prompt | llm | StrOutputParser()

            try:
                response = rag_chain.invoke({"context": formatted_context, "question": query_text})
                st.subheader("Respuesta Generada")
                st.markdown(response)
                
                # Guardar la respuesta y contexto para el feedback
                st.session_state.last_response_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": query_text,
                    "answer": response,
                    "context": formatted_context
                }

                with st.expander("Ver fuentes utilizadas"):
                    st.markdown(formatted_context)

            except Exception as e:
                st.error(f"Ocurrió un error al generar la respuesta: {e}")

    # --- Sistema de Feedback ---
    if st.session_state.last_response:
        if not st.session_state.feedback_given:
            st.write("\n---")
            col1, col2 = st.columns([1, 10])
            with col1:
                if st.button("👍"):
                    log_feedback(rating="👍 Buena respuesta")
                    st.session_state.feedback_given = True
                    st.rerun()
            with col2:
                if st.button("👎"):
                    log_feedback(rating="👎 Mala respuesta")
                    st.session_state.feedback_given = True
                    st.rerun()
        else:
            st.success("¡Gracias por tu feedback!")

if __name__ == '__main__':
    main()
