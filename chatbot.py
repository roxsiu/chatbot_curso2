"""
Chatbot RAG para DMC Institute
Integra búsqueda semántica (ChromaDB) con generación de respuestas (OpenAI LLM).
Implementa manejo básico de alucinaciones y seguridad.
"""
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
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

        # Validar API Key
        if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY == "tu-api-key-aqui":
            raise ValueError(
                "API Key no configurada. Edita .env con tu OPENAI_API_KEY"
            )

        # 1. Indexador de documentos (embeddings + ChromaDB)
        self.indexer = DocumentIndexer()

        # 2. Modelo LLM para generación de respuestas
        self.llm = ChatOpenAI(
            model_name=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            openai_api_key=Config.OPENAI_API_KEY
        )

        # 3. Memoria de conversación
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # 4. Chain de RAG (se crea después de indexar)
        self.chain = None

        # 5. Historial para mostrar
        self.history = []

        print("✓ Chatbot DMC inicializado\n")

    def setup(self, pdf_folder: str = None, force_reindex: bool = False):
        """
        Configura el chatbot: indexa documentos y crea la chain de RAG.

        Args:
            pdf_folder: Carpeta con PDFs
            force_reindex: Si True, re-indexa aunque ya exista
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

        # Crear retriever a partir del vector store
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Config.DEFAULT_TOP_K}
        )

        # Prompt personalizado para el QA
        qa_prompt = PromptTemplate(
            template="""
{system_prompt}

Contexto (información de documentos de DMC):
{context}

Historial de conversación:
{chat_history}

Pregunta del usuario: {question}

Respuesta:""",
            input_variables=["context", "chat_history", "question"],
            partial_variables={"system_prompt": Config.SYSTEM_PROMPT}
        )

        # Crear chain de RAG conversacional
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=False
        )

        print("✓ Chain RAG configurada y lista\n")

    def chat(self, user_message: str) -> Dict:
        """
        Procesa un mensaje del usuario y genera una respuesta.

        Args:
            user_message: Mensaje/pregunta del usuario

        Returns:
            Diccionario con respuesta, fuentes y metadata
        """
        if not self.chain:
            return {
                "answer": "❌ El chatbot no está configurado. Ejecuta setup() primero.",
                "sources": [],
                "error": True
            }

        # Validación básica de seguridad (anti-inyección)
        if self._is_unsafe_input(user_message):
            return {
                "answer": "No puedo procesar esa solicitud. ¿Puedo ayudarte con información sobre los programas de DMC?",
                "sources": [],
                "blocked": True
            }

        try:
            # Ejecutar la chain RAG
            result = self.chain.invoke({"question": user_message})

            answer = result.get("answer", "No pude generar una respuesta.")
            source_docs = result.get("source_documents", [])

            # Extraer fuentes
            sources = []
            for doc in source_docs:
                source_info = {
                    "archivo": doc.metadata.get("source", "N/A"),
                    "pagina": doc.metadata.get("page", "N/A"),
                    "texto_preview": doc.page_content[:150] + "..."
                }
                sources.append(source_info)

            # Validación de alucinaciones básica
            answer = self._validate_response(answer, source_docs)

            # Guardar en historial
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
            error_msg = f"Error al procesar la pregunta: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "answer": "Ocurrió un error al procesar tu pregunta. Intenta de nuevo.",
                "sources": [],
                "error": True,
                "error_detail": str(e)
            }

    def _is_unsafe_input(self, text: str) -> bool:
        """
        Validación básica de seguridad contra inyecciones de prompt.

        Args:
            text: Texto del usuario

        Returns:
            True si el input parece malicioso
        """
        # Patrones sospechosos
        unsafe_patterns = [
            "ignora las instrucciones",
            "ignore your instructions",
            "olvida tu rol",
            "forget your role",
            "actúa como",
            "act as",
            "eres ahora",
            "you are now",
            "nuevo prompt",
            "new prompt",
            "system prompt",
        ]

        text_lower = text.lower().strip()

        for pattern in unsafe_patterns:
            if pattern in text_lower:
                print(f"⚠ Input bloqueado por seguridad: patrón '{pattern}'")
                return True

        return False

    def _validate_response(self, answer: str, source_docs: list) -> str:
        """
        Validación básica contra alucinaciones.
        Verifica que la respuesta tenga relación con los documentos fuente.

        Args:
            answer: Respuesta generada por el LLM
            source_docs: Documentos fuente usados

        Returns:
            Respuesta validada
        """
        # Si no hay documentos fuente, advertir
        if not source_docs:
            return (
                "No encontré información relevante en mis documentos para "
                "responder tu pregunta. ¿Puedo ayudarte con algo más sobre "
                "los programas de DMC Institute?"
            )

        # Si la respuesta es muy corta, puede ser un fallo
        if len(answer.strip()) < 10:
            return (
                "No pude generar una respuesta adecuada. "
                "¿Podrías reformular tu pregunta?"
            )

        return answer

    def get_history(self) -> List[Dict]:
        """Retorna el historial de conversación"""
        return self.history

    def clear_history(self):
        """Limpia el historial y la memoria"""
        self.history = []
        self.memory.clear()
        print("✓ Historial limpiado")

    def get_stats(self) -> Dict:
        """Retorna estadísticas del chatbot"""
        indexer_stats = self.indexer.get_stats()
        return {
            "llm_model": Config.LLM_MODEL,
            "temperature": Config.LLM_TEMPERATURE,
            "messages_count": len(self.history),
            "memory_messages": len(self.memory.chat_memory.messages),
            **indexer_stats
        }


# ============================================================================
# DEMO / PRUEBA
# ============================================================================

def main():
    """Demo interactiva del chatbot"""
    Config.print_config()

    # Crear chatbot
    bot = ChatbotDMC()

    # Configurar (indexar documentos)
    bot.setup()

    # Modo interactivo
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

        # Mostrar respuesta
        print(f"\nBot: {response['answer']}")

        # Mostrar fuentes (opcional)
        if response.get('sources'):
            print(f"\n  📚 Fuentes consultadas: {len(response['sources'])}")
            for s in response['sources'][:2]:  # Mostrar máximo 2 fuentes
                import os
                archivo = os.path.basename(s['archivo'])
                print(f"     - {archivo} (pág. {s['pagina']})")

        print()


if __name__ == "__main__":
    main()
