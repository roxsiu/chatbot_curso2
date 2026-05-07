"""
Configuración del Chatbot DMC
Variables de entorno y parámetros generales
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuración centralizada del proyecto"""

    # =========================================================================
    # API Keys
    # =========================================================================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "tu-api-key-aqui")

    # =========================================================================
    # Modelos
    # =========================================================================
    # Modelo de embeddings
    EMBEDDING_MODEL = "text-embedding-3-small"

    # Modelo LLM para generación de respuestas
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 1000

    # =========================================================================
    # ChromaDB (Vector Store)
    # =========================================================================
    CHROMA_COLLECTION_NAME = "dmc_documentos"
    CHROMA_PERSIST_DIR = "./chroma_db"

    # =========================================================================
    # Procesamiento de documentos
    # =========================================================================
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    PDF_FOLDER = "./pdfs"

    # =========================================================================
    # Búsqueda
    # =========================================================================
    DEFAULT_TOP_K = 5

    # =========================================================================
    # Prompt del sistema
    # =========================================================================
    SYSTEM_PROMPT = """Eres un asistente virtual de DMC Institute, una institución educativa 
especializada en programas de Data & Analytics e Inteligencia Artificial.

Tu rol es responder preguntas sobre los programas, diplomas y cursos que ofrece DMC Institute.

Reglas:
1. Responde SOLO con información que encuentres en el contexto proporcionado.
2. Si no encuentras la información en el contexto, di: "No tengo información sobre eso en mis documentos. ¿Puedo ayudarte con algo más sobre nuestros programas?"
3. Sé amable, profesional y conciso.
4. Responde en español.
5. Si te preguntan algo que no tiene que ver con DMC o sus programas, redirige la conversación amablemente.
"""

    @classmethod
    def print_config(cls):
        """Muestra la configuración actual"""
        print("=" * 60)
        print("CONFIGURACIÓN DEL CHATBOT DMC")
        print("=" * 60)
        print(f"  Embedding Model:  {cls.EMBEDDING_MODEL}")
        print(f"  LLM Model:        {cls.LLM_MODEL}")
        print(f"  Temperature:      {cls.LLM_TEMPERATURE}")
        print(f"  Chunk Size:       {cls.CHUNK_SIZE}")
        print(f"  Chunk Overlap:    {cls.CHUNK_OVERLAP}")
        print(f"  Top K:            {cls.DEFAULT_TOP_K}")
        print(f"  ChromaDB Dir:     {cls.CHROMA_PERSIST_DIR}")
        print(f"  PDF Folder:       {cls.PDF_FOLDER}")
        api_status = "✓ Configurada" if cls.OPENAI_API_KEY != "tu-api-key-aqui" else "✗ No configurada"
        print(f"  OpenAI API Key:   {api_status}")
        print("=" * 60 + "\n")
