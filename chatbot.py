"""
Chatbot RAG para DMC Institute
Usa LCEL (LangChain Expression Language) con paquetes modulares.
Compatible con LangChain v1.0+
"""
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from indexer import DocumentIndexer
from config import Config
import warnings

warnings.filterwarnings('ignore')


class ChatbotDMC:
    """
    Chatbot inteligente para DMC Institute.
    Usa RAG (Retrieval-Augmented Generation) para responder
    preguntas basándose en documentos reales de DMC.
    """

    def __init__(self):
        """Inicializa el chatbot con todos sus componentes"""
        print("=" * 60)
        print("INICIALIZANDO CHATBOT DMC")
        print("=" * 60 + "\n")

        if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY == "tu-api-key-aqui":
            raise ValueError("API Key no configurada. Edita .env con tu OPENAI_API_KEY")

        # 1. Indexador de documentos
        self.indexer = DocumentIndexer()

        # 2. Modelo LLM
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            openai_api_key=Config.OPENAI_API_KEY
        )

        # 3. Historial de conversación (lista de mensajes)
        self.chat_history: List = []

        # 4. Retriever y chain (se crean después de indexar)
        self.retriever = None
        self.chain = None

        # 5. Historial para mostrar
        self.history = []

        print("✓ Chatbot DMC inicializado\n")

    def setup(self, pdf_folder: str = None, force_reindex: bool = False):
        """
        Configura el chatbot: indexa documentos y crea la chain RAG.
        """
        # Intentar cargar vector store existente
        vector_store = None
        if not force_reindex:
            vector_store = self.indexer.load_existing_store()

        # Si no existe, indexar
        if not vector_store:
            vector_store = self.indexer.index_pdfs(pdf_folder)

        if not vector_store:
            raise RuntimeError("No se pudo crear el vector store")

        # Crear retriever
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Config.DEFAULT_TOP_K}
        )

        # Crear prompt RAG
        prompt = ChatPromptTemplate.from_messages([
            ("system", Config.SYSTEM_PROMPT + "\n\nContexto de documentos DMC:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # Función para formatear los documentos recuperados
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Crear chain con LCEL
        self.chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(self.retriever.invoke(x["question"]))
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

        print("✓ Chain RAG configurada y lista\n")

    def chat(self, user_message: str) -> Dict:
        """
        Procesa un mensaje del usuario y genera una respuesta.
        """
        if not self.chain:
            return {
                "answer": "❌ El chatbot no está configurado. Ejecuta setup() primero.",
                "sources": [],
                "error": True
            }

        # Validación de seguridad
        if self._is_unsafe_input(user_message):
            return {
                "answer": "No puedo procesar esa solicitud. ¿Puedo ayudarte con información sobre los programas de DMC?",
                "sources": [],
                "blocked": True
            }

        try:
            # Obtener documentos fuente para mostrar
            source_docs = self.retriever.invoke(user_message)

            # Ejecutar la chain RAG
            answer = self.chain.invoke({
                "question": user_message,
                "chat_history": self.chat_history
            })

            # Validación de alucinaciones
            if not source_docs:
                answer = (
                    "No encontré información relevante en mis documentos para "
                    "responder tu pregunta. ¿Puedo ayudarte con algo más sobre "
                    "los programas de DMC Institute?"
                )

            # Extraer fuentes
            sources = []
            for doc in source_docs:
                sources.append({
                    "archivo": doc.metadata.get("source", "N/A"),
                    "pagina": doc.metadata.get("page", "N/A"),
                    "texto_preview": doc.page_content[:150] + "..."
                })

            # Actualizar historial de conversación
            self.chat_history.append(HumanMessage(content=user_message))
            self.chat_history.append(AIMessage(content=answer))

            # Guardar en historial legible
            self.history.append({
                "user": user_message,
                "assistant": answer
            })

            return {
                "answer": answer,
                "sources": sources,
                "error": False
            }

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {
                "answer": "Ocurrió un error al procesar tu pregunta. Intenta de nuevo.",
                "sources": [],
                "error": True,
                "error_detail": str(e)
            }

    def _is_unsafe_input(self, text: str) -> bool:
        """Validación básica contra inyecciones de prompt"""
        unsafe_patterns = [
            "ignora las instrucciones", "ignore your instructions",
            "olvida tu rol", "forget your role",
            "actúa como", "act as",
            "eres ahora", "you are now",
            "nuevo prompt", "new prompt",
            "system prompt",
        ]
        text_lower = text.lower().strip()
        for pattern in unsafe_patterns:
            if pattern in text_lower:
                print(f"⚠ Input bloqueado: patrón '{pattern}'")
                return True
        return False

    def get_history(self) -> List[Dict]:
        """Retorna el historial de conversación"""
        return self.history

    def clear_history(self):
        """Limpia el historial y la memoria"""
        self.history = []
        self.chat_history = []
        print("✓ Historial limpiado")

    def get_stats(self) -> Dict:
        """Retorna estadísticas del chatbot"""
        indexer_stats = self.indexer.get_stats()
        return {
            "llm_model": Config.LLM_MODEL,
            "temperature": Config.LLM_TEMPERATURE,
            "messages_count": len(self.history),
            **indexer_stats
        }


# ============================================================================
# MODO INTERACTIVO
# ============================================================================

def main():
    """Demo interactiva del chatbot"""
    Config.print_config()

    bot = ChatbotDMC()
    bot.setup()

    print("=" * 60)
    print("CHATBOT DMC - Modo Interactivo")
    print("=" * 60)
    print("Escribe tu pregunta sobre DMC Institute.")
    print("Escribe 'salir' para terminar.")
    print("Escribe 'historial' para ver el historial.")
    print("Escribe 'stats' para ver estadísticas.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n¡Hasta luego!")
            break

        if not user_input:
            continue

        if user_input.lower() == "salir":
            print("\n¡Gracias por usar el Chatbot de DMC! ¡Hasta pronto!")
            break

        if user_input.lower() == "historial":
            history = bot.get_history()
            if not history:
                print("No hay historial aún.\n")
            else:
                for i, h in enumerate(history, 1):
                    print(f"\n--- Mensaje {i} ---")
                    print(f"Tú: {h['user']}")
                    print(f"Bot: {h['assistant']}")
            print()
            continue

        if user_input.lower() == "stats":
            stats = bot.get_stats()
            print("\nEstadísticas:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
            print()
            continue

        # Enviar mensaje al chatbot
        response = bot.chat(user_input)
        print(f"\nBot: {response['answer']}")

        if response.get('sources'):
            import os
            print(f"\n  📚 Fuentes consultadas: {len(response['sources'])}")
            for s in response['sources'][:2]:
                archivo = os.path.basename(s['archivo'])
                print(f"     - {archivo} (pág. {s['pagina']})")
        print()


if __name__ == "__main__":
    main()
