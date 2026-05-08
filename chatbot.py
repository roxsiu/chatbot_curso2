"""
Módulo de Indexación de Documentos
Carga PDFs, los divide en chunks, genera embeddings y los almacena en ChromaDB.
Usa paquetes modulares de LangChain (compatibles con v1.0+).
"""
from typing import List, Dict
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import Config
import warnings
import os
import shutil

warnings.filterwarnings('ignore')


class DocumentIndexer:
    """
    Indexador de documentos para el chatbot de DMC.
    Carga PDFs, los procesa en chunks y los almacena en ChromaDB.
    """

    def __init__(self):
        """Inicializa el indexador con embeddings de OpenAI"""
        print("Inicializando indexador de documentos...")

        if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY == "tu-api-key-aqui":
            raise ValueError(
                "API Key de OpenAI no configurada. "
                "Edita el archivo .env y agrega tu OPENAI_API_KEY"
            )

        # Modelo de embeddings
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )

        self.vector_store = None

        print(f"✓ Embeddings: {Config.EMBEDDING_MODEL}")
        print(f"✓ Chunk size: {Config.CHUNK_SIZE}, Overlap: {Config.CHUNK_OVERLAP}")
        print("✓ Indexador inicializado\n")

    def load_pdfs(self, pdf_folder: str = None) -> list:
        """Carga todos los PDFs de una carpeta"""
        folder = pdf_folder or Config.PDF_FOLDER

        if not os.path.exists(folder):
            print(f"❌ La carpeta '{folder}' no existe")
            return []

        print(f"Cargando PDFs desde: {folder}")

        loader = PyPDFDirectoryLoader(folder)
        documents = loader.load()

        if not documents:
            print("❌ No se encontraron PDFs o no tienen texto")
            return []

        print(f"✓ {len(documents)} páginas cargadas")

        archivos = set()
        for doc in documents:
            source = doc.metadata.get("source", "desconocido")
            archivos.add(os.path.basename(source))
        print(f"✓ Archivos: {', '.join(archivos)}")

        return documents

    def split_documents(self, documents: list) -> list:
        """Divide documentos en chunks"""
        print(f"\nDividiendo {len(documents)} documentos en chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"✓ {len(chunks)} chunks creados")
        return chunks

    def create_vector_store(self, chunks: list) -> Chroma:
        """Crea el vector store en ChromaDB"""
        print(f"\nCreando vector store en ChromaDB...")

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=Config.CHROMA_COLLECTION_NAME,
            persist_directory=Config.CHROMA_PERSIST_DIR
        )

        print(f"✓ Vector store creado con {len(chunks)} documentos")
        return self.vector_store

    def index_pdfs(self, pdf_folder: str = None) -> Chroma:
        """Pipeline completo: Carga PDFs → Chunks → ChromaDB"""
        print("=" * 60)
        print("INDEXACIÓN DE DOCUMENTOS")
        print("=" * 60 + "\n")

        documents = self.load_pdfs(pdf_folder)
        if not documents:
            return None

        chunks = self.split_documents(documents)
        vector_store = self.create_vector_store(chunks)

        print(f"\n{'=' * 60}")
        print("✓ INDEXACIÓN COMPLETADA")
        print(f"{'=' * 60}\n")

        return vector_store

    def load_existing_store(self) -> Chroma:
        """Carga un vector store existente desde disco"""
        if not os.path.exists(Config.CHROMA_PERSIST_DIR):
            return None

        print("Cargando vector store existente...")

        self.vector_store = Chroma(
            collection_name=Config.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR
        )

        count = self.vector_store._collection.count()
        if count == 0:
            return None

        print(f"✓ Vector store cargado: {count} documentos")
        return self.vector_store

    def search(self, query: str, k: int = None) -> List[Dict]:
        """Búsqueda semántica en el vector store"""
        if not self.vector_store:
            print("❌ No hay vector store. Ejecuta index_pdfs() primero.")
            return []

        k = k or Config.DEFAULT_TOP_K
        results = self.vector_store.similarity_search_with_score(query=query, k=k)

        formatted = []
        for doc, score in results:
            formatted.append({
                "text": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata
            })
        return formatted

    def delete_store(self):
        """Elimina el vector store del disco"""
        if os.path.exists(Config.CHROMA_PERSIST_DIR):
            shutil.rmtree(Config.CHROMA_PERSIST_DIR)
            print(f"✓ Vector store eliminado")
        self.vector_store = None

    def get_stats(self) -> Dict:
        """Retorna estadísticas del vector store"""
        if not self.vector_store:
            return {"status": "No inicializado", "documents": 0}
        count = self.vector_store._collection.count()
        return {
            "status": "Activo",
            "documents": count,
            "collection": Config.CHROMA_COLLECTION_NAME,
            "embedding_model": Config.EMBEDDING_MODEL
        }


def main():
    """Demo de indexación y búsqueda"""
    Config.print_config()

    indexer = DocumentIndexer()
    vector_store = indexer.index_pdfs()

    if not vector_store:
        print("No se pudo crear el vector store")
        return

    queries = [
        "¿Qué programas ofrece DMC?",
        "¿Cuáles son los requisitos para el diploma?",
        "¿Qué tecnologías se enseñan?",
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print("=" * 60)

        results = indexer.search(query, k=3)
        for i, r in enumerate(results, 1):
            text_preview = r['text'][:200].replace('\n', ' ')
            print(f"\n{i}. [Score: {r['score']:.4f}]")
            print(f"   Fuente: {os.path.basename(r['metadata'].get('source', 'N/A'))}")
            print(f"   Texto: {text_preview}...")

    print("\n✓ Demo completada")


if __name__ == "__main__":
    main()
