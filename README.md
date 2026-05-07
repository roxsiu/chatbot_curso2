
# Chatbot DMC Institute - LLM + RAG

## Descripción

Chatbot inteligente para **DMC Institute** que responde preguntas sobre los programas y diplomas de la institución, utilizando **RAG (Retrieval-Augmented Generation)**.

El chatbot carga documentos PDF de DMC (brochures de los diplomas), los indexa en una base de datos vectorial y usa un LLM para generar respuestas basadas en información real.

## Arquitectura

```
Usuario → Validación de seguridad → Búsqueda semántica (ChromaDB) → LLM (OpenAI) → Respuesta
```

### Componentes principales

| Componente | Tecnología | Descripción |
|---|---|---|
| Framework | LangChain | Orquestación de la pipeline RAG |
| Embeddings | OpenAI (text-embedding-3-small) | Vectorización de textos |
| Vector Store | ChromaDB | Almacenamiento y búsqueda de vectores |
| LLM | OpenAI (gpt-4o-mini) | Generación de respuestas |
| Carga de PDFs | PyPDF + LangChain | Extracción de texto de documentos |

### Diagrama de arquitectura

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   PDFs DMC  │────▶│  Text Split  │────▶│  Embeddings  │
│  (brochures)│     │   (chunks)   │     │   (OpenAI)   │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Respuesta  │◀────│   LLM Chat   │◀────│   ChromaDB   │
│  al usuario │     │ (gpt-4o-mini)│     │ (vector DB)  │
└─────────────┘     └──────────────┘     └──────────────┘
```

## Requisitos

- Python 3.9+
- API Key de OpenAI

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/chatbot-dmc.git
cd chatbot-dmc
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar API Key

Edita el archivo `.env`:

```bash
OPENAI_API_KEY=tu-api-key-real-aqui
```

### 4. Colocar PDFs

Coloca los brochures de DMC en la carpeta `pdfs/`:

```
pdfs/
├── brochure_diploma_AIE_2.pdf
└── brochure_diploma_DE_6.pdf
```

## Ejecución

### Opción 1: Script interactivo

```bash
python chatbot.py
```

### Opción 2: Notebook (Jupyter / Google Colab)

Abre `chatbot_dmc.ipynb` y ejecuta celda por celda.

### Opción 3: Script de pruebas

```bash
python test_chatbot.py
```

### Opción 4: Solo indexación (sin chatbot)

```bash
python indexer.py
```

## Estructura del proyecto

```
chatbot-dmc/
├── config.py              # Configuración centralizada
├── indexer.py             # Módulo de indexación (PDFs → ChromaDB)
├── chatbot.py             # Chatbot RAG completo (interactivo)
├── test_chatbot.py        # Script de pruebas automatizadas
├── chatbot_dmc.ipynb      # Notebook para Jupyter/Colab
├── requirements.txt       # Dependencias Python
├── .env                   # Variables de entorno (API Key)
├── README.md              # Esta documentación
└── pdfs/                  # Carpeta con PDFs de DMC
    ├── brochure_diploma_AIE_2.pdf
    └── brochure_diploma_DE_6.pdf
```

## Funcionalidades

- **RAG**: Respuestas basadas en documentos reales de DMC
- **Memoria**: Mantiene contexto durante la conversación
- **Seguridad**: Detecta intentos de inyección de prompt
- **Anti-alucinaciones**: Valida que las respuestas tengan fuentes
- **Fuentes**: Muestra los documentos consultados para cada respuesta

## Tecnologías

- Python 3.9+
- LangChain
- OpenAI (GPT-4o-mini + text-embedding-3-small)
- ChromaDB
- PyPDF

