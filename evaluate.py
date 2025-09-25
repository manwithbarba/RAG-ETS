import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import os
import asyncio

# Importar la lógica de la aplicación principal
# Asegúrate de que app.py esté en el mismo directorio
from app import load_resources, get_rag_chain

# --- Configuración ---
EVAL_DATASET_PATH = "eval_dataset.csv"

def run_rag_pipeline(query, db, llm, k=3):
    """
    Ejecuta el pipeline RAG para una sola pregunta y devuelve los resultados.
    """
    retriever = db.as_retriever(search_kwargs={'k': k})
    rag_chain = get_rag_chain(llm, retriever)
    
    retrieved_docs = retriever.invoke(query)
    
    formatted_context = ""
    contexts = []
    for i, doc in enumerate(retrieved_docs):
        contexts.append(doc.page_content)
        formatted_context += f"### Fuente [{i+1}]\n"
        formatted_context += f"Documento: {doc.metadata.get('source', 'N/A').replace('documentos\', '')}\n"
        formatted_context += f"Contenido: {doc.page_content}\n\n"
        
    answer = rag_chain.invoke({"context": formatted_context, "question": query})
    
    return {
        "answer": answer,
        "contexts": contexts
    }

def main():
    """
    Función principal para ejecutar la evaluación.
    """
    # --- 1. Cargar Recursos ---
    print("Cargando recursos (modelos y base de datos)...")
    db, llm, error_message = load_resources()
    if error_message:
        print(f"Error al cargar recursos: {error_message}")
        return

    # --- 2. Cargar el Golden Set ---
    print(f"Cargando el Golden Set desde {EVAL_DATASET_PATH}...")
    if not os.path.exists(EVAL_DATASET_PATH):
        print(f"Error: No se encontró el archivo '{EVAL_DATASET_PATH}'. Por favor, créalo con las columnas ['question', 'ground_truth_answer', 'ground_truth_context'].")
        return
        
    eval_df = pd.read_csv(EVAL_DATASET_PATH)
    eval_dataset = Dataset.from_pandas(eval_df)

    # --- 3. Ejecutar el Pipeline para cada Pregunta ---
    print("Ejecutando el pipeline RAG para cada pregunta del Golden Set...")
    results = []
    for entry in eval_dataset:
        result = run_rag_pipeline(entry['question'], db, llm)
        results.append({
            "question": entry['question'],
            "ground_truth": entry['ground_truth_answer'], # Ragas espera esta columna
            "answer": result['answer'],
            "contexts": result['contexts']
        })
    
    results_dataset = Dataset.from_list(results)

    # --- 4. Ejecutar la Evaluación con Ragas ---
    print("Calculando las métricas de evaluación con Ragas... (Esto puede tardar)")
    
    # Configurar las métricas a utilizar
    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ]
    
    # Ejecutar la evaluación
    # Ragas usa un LLM de OpenAI por defecto como juez. Se puede configurar para usar otro.
    # Por simplicidad, usamos el default. Requiere una API key de OpenAI configurada como variable de entorno.
    if 'OPENAI_API_KEY' not in os.environ:
        print("\nADVERTENCIA: La variable de entorno OPENAI_API_KEY no está configurada.")
        print("Ragas usará un LLM de OpenAI por defecto para juzgar los resultados.")
        print("Sin esta clave, la evaluación fallará. Por favor, configúrala para continuar.")
        return

    result = evaluate(results_dataset, metrics)
    
    # --- 5. Mostrar Resultados ---
    print("\n--- INFORME DE EVALUACIÓN RAG ---")
    print(result)
    print("-------------------------------------")

if __name__ == "__main__":
    # Ragas usa asyncio, y puede dar problemas en algunos entornos de Windows.
    # Este workaround suele solucionarlo.
    asyncio.run(main())
