"""
Módulo de Indexación de Documentos
Carga PDFs, los divide en chunks, genera embeddings y los almacena en ChromaDB.
"""
from typing import List, Dict
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
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

        # Validar API Key
        if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY == "tu-api-key-aqui":
            raise ValueError(
                "API Key de OpenAI no configurada. "
                "Edita el archivo .env y agrega tu OPENAI_API_KEY"
            )

        # Crear modelo de embeddings
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )

        # Text splitter para dividir documentos en chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )

        # Vector store (se crea al indexar)
        self.vector_store = None

        print(f"✓ Embeddings: {Config.EMBEDDING_MODEL}")
        print(f"✓ Chunk size: {Config.CHUNK_SIZE}, Overlap: {Config.CHUNK_OVERLAP}")
        print("✓ Indexador inicializado\n")

    def load_pdfs(self, pdf_folder: str = None) -> list:
        """
        Carga todos los PDFs de una carpeta usando LangChain

        Args:
            pdf_folder: Ruta de la carpeta con PDFs

        Returns:
            Lista de documentos cargados
        """
        folder = pdf_folder or Config.PDF_FOLDER

        if not os.path.exists(folder):
            print(f"❌ La carpeta '{folder}' no existe")
            return []

        print(f"Cargando PDFs desde: {folder}")

        # Usar PyPDFDirectoryLoader de LangChain
        loader = PyPDFDirectoryLoader(folder)
        documents = loader.load()

        if not documents:
            print("❌ No se encontraron PDFs o no tienen texto")
            return []

        print(f"✓ {len(documents)} páginas cargadas")

        # Mostrar resumen de archivos
        archivos = set()
        for doc in documents:
            source = doc.metadata.get("source", "desconocido")
            archivos.add(os.path.basename(source))

        print(f"✓ Archivos encontrados: {', '.join(archivos)}")

        return documents

    def split_documents(self, documents: list) -> list:
        """
        Divide documentos en chunks más pequeños

        Args:
            documents: Lista de documentos de LangChain

        Returns:
            Lista de chunks
        """
        print(f"\nDividiendo {len(documents)} documentos en chunks...")

        chunks = self.text_splitter.split_documents(documents)

        print(f"✓ {len(chunks)} chunks creados")

        return chunks

    def create_vector_store(self, chunks: list, persist: bool = True) -> Chroma:
        """
        Crea el vector store en ChromaDB

        Args:
            chunks: Lista de chunks de documentos
            persist: Si True, guarda en disco

        Returns:
            Vector store de ChromaDB
        """
        print(f"\nCreando vector store en ChromaDB...")
        print(f"  Colección: {Config.CHROMA_COLLECTION_NAME}")

        # Crear ChromaDB con los documentos
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=Config.CHROMA_COLLECTION_NAME,
            persist_directory=Config.CHROMA_PERSIST_DIR if persist else None
        )

        print(f"✓ Vector store creado con {len(chunks)} documentos")
        print(f"✓ Persistido en: {Config.CHROMA_PERSIST_DIR}")

        return self.vector_store

    def index_pdfs(self, pdf_folder: str = None) -> Chroma:
        """
        Pipeline completo: Carga PDFs → Chunks → ChromaDB

        Args:
            pdf_folder: Carpeta con PDFs (usa Config si no se especifica)

        Returns:
            Vector store creado
        """
        print("=" * 60)
        print("INDEXACIÓN DE DOCUMENTOS")
        print("=" * 60 + "\n")

        # Paso 1: Cargar PDFs
        documents = self.load_pdfs(pdf_folder)
        if not documents:
            return None

        # Paso 2: Dividir en chunks
        chunks = self.split_documents(documents)

        # Paso 3: Crear vector store
        vector_store = self.create_vector_store(chunks)

        print(f"\n{'=' * 60}")
        print("✓ INDEXACIÓN COMPLETADA")
        print(f"{'=' * 60}\n")

        return vector_store

    def load_existing_store(self) -> Chroma:
        """
        Carga un vector store existente desde disco

        Returns:
            Vector store cargado o None si no existe
        """
        if not os.path.exists(Config.CHROMA_PERSIST_DIR):
            print("No se encontró un vector store existente")
            return None

        print("Cargando vector store existente...")

        self.vector_store = Chroma(
            collection_name=Config.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR
        )

        # Verificar que tiene documentos
        count = self.vector_store._collection.count()
        if count == 0:
            print("Vector store vacío, necesita re-indexar")
            return None

        print(f"✓ Vector store cargado: {count} documentos")
        return self.vector_store

    def search(self, query: str, k: int = None) -> List[Dict]:
        """
        Búsqueda semántica en el vector store

        Args:
            query: Texto de búsqueda
            k: Número de resultados

        Returns:
            Lista de resultados con texto, score y metadata
        """
        if not self.vector_store:
            print("❌ No hay vector store. Ejecuta index_pdfs() primero.")
            return []

        k = k or Config.DEFAULT_TOP_K

        # Búsqueda por similitud con scores
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k
        )

        # Formatear resultados
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
            print(f"✓ Vector store eliminado: {Config.CHROMA_PERSIST_DIR}")
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
            "persist_dir": Config.CHROMA_PERSIST_DIR,
            "embedding_model": Config.EMBEDDING_MODEL
        }


# ============================================================================
# DEMO / PRUEBA
# ============================================================================

def main():
    """Demo de indexación y búsqueda"""
    Config.print_config()

    # Crear indexador
    indexer = DocumentIndexer()

    # Indexar PDFs
    vector_store = indexer.index_pdfs()

    if not vector_store:
        print("No se pudo crear el vector store")
        return

    # Pruebas de búsqueda
    queries = [
        "¿Qué programas ofrece DMC?",
        "¿Cuáles son los requisitos para el diploma?",
        "¿Qué es un chatbot?",
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

    # Estadísticas
    print(f"\n{'=' * 60}")
    print("ESTADÍSTICAS")
    print("=" * 60)
    for k, v in indexer.get_stats().items():
        print(f"  {k}: {v}")

    print("\n✓ Demo completada")


if __name__ == "__main__":
    main()
