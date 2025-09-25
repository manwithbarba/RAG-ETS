import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Definir la carpeta de documentos y la base de datos vectorial
DATA_PATH = "documentos/"
DB_PATH = "vectorstores/db/"

def create_vector_db():
    """
    Carga documentos desde la carpeta 'documentos', los procesa en fragmentos (chunks),
    genera embeddings para cada fragmento y los almacena en una base de datos vectorial
    ChromaDB que persiste en disco.
    """
    # Configurar loaders para diferentes tipos de archivo (PDF y DOCX)
    pdf_loader = DirectoryLoader(DATA_PATH, glob='**/*.pdf', loader_cls=PyPDFLoader, show_progress=True)
    docx_loader = DirectoryLoader(DATA_PATH, glob='**/*.docx', loader_cls=Docx2txtLoader, show_progress=True)
    
    print(f"Cargando documentos desde {DATA_PATH}...")
    documents = pdf_loader.load() + docx_loader.load()
    print(f"Cargados {len(documents)} documentos.")

    # Segmentar los documentos en fragmentos más pequeños
    # La estrategia de chunking (tamaño y superposición) es clave para la calidad del RAG.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Segmentados {len(documents)} documentos en {len(texts)} chunks.")

    # Generar embeddings para cada fragmento usando un modelo de Hugging Face
    print("Generando embeddings... (Esto puede tardar un momento)")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Crear y persistir la base de datos vectorial usando ChromaDB
    # Los embeddings y los textos se almacenan para su posterior recuperación.
    db = Chroma.from_documents(texts, embeddings, persist_directory=DB_PATH)
    print(f"Guardada la base de datos de vectores en {DB_PATH}")

if __name__ == "__main__":
    print("Iniciando proceso de indexación...")
    create_vector_db()
    print("Proceso de indexación finalizado.")

