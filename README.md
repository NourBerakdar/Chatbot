# 🧠 RAG Chatbot with Ollama, ChromaDB & Gradio

A lightweight **Retrieval-Augmented Generation (RAG) chatbot** powered by **Ollama**, **ChromaDB**, and **Gradio**. This chatbot ingests PDF files, creates vector embeddings, and enables natural language Q&A with a fallback to web search when necessary.

---

## 📖 Overview

This project integrates **Ollama** for large language models and embeddings, **ChromaDB** as the vector store, and a **Gradio**-based interface to provide an interactive chatbot capable of:

- Answering user queries based on custom PDF knowledge bases.
- Falling back to DuckDuckGo search if relevant information is not found locally.
- Being extended to other use cases such as internal documentation, product manuals, and FAQs.

---

## 🧩 Why RAG?

**Retrieval-Augmented Generation (RAG)** enhances this chatbot by combining two essential capabilities: document retrieval and generative response. This approach addresses key limitations found in traditional LLM deployments:

- **Domain-specific responses**: By retrieving content directly from your ingested PDFs, the chatbot delivers precise, contextualized answers.
- **Improved accuracy**: Leveraging ChromaDB reduces hallucinations and enhances reliability by grounding responses in your data.
- **Balanced retrieval and generation**: When documents fall short, the system performs a web search to provide additional relevant information.

This makes the chatbot especially useful for customer service, internal support, and any application requiring trustworthy, source-backed answers.

---

## 📂 Repository Structure

```
📦 Chatbot
 ┣ 📁 data/                 # Directory for PDF documents to ingest
 ┣ 📁 chroma_db/            # Auto-generated vector database
 ┣ 📄 ingest_data.py        # PDF ingestion and embedding pipeline
 ┣ 📄 chatbot.py            # Chatbot application with UI and fallback search
 ┣ 📄 config.py             # Configuration file for models and settings
 ┣ 📄 requirements.txt      # Project dependencies
 ┣ 📄 demo.gif              # (optional) Demo animation of the chatbot
 ┗ 📄 README.md             # Documentation file
```

---

## ✨ Features

- 🧠 **RAG pipeline** with Ollama embeddings and LLMs  
- 📄 **PDF ingestion** into ChromaDB for vector storage  
- 🌍 **DuckDuckGo fallback search** for external queries  
- 💬 **Gradio interface** with streaming chatbot responses  
- ⚙️ **Modular and configurable** system via `config.py`

---

## ✅ Prerequisites

- Python 3.9+
- Ollama installed and accessible via `ollama run`
- PDF documents placed inside the `/data` directory

---

## ⚡ Installation & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/NourBerakdar/Chatbot.git
cd Chatbot
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Ingest PDF documents
Add your PDFs to the `/data` folder and run:
```bash
python ingest_data.py
```

### 4️⃣ Launch the chatbot
```bash
python chatbot.py
```

Open the Gradio link provided in the terminal (e.g., `http://127.0.0.1:7860`).

---

## ⚙️ Configuration

Customize your chatbot via `config.py`:

```python
# Paths
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

# Model settings
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# Text splitting
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100

# Retriever
NUM_RESULTS = 3
```

---

## 🎥 Demo

![Demo](demo.gif)

---

## 🛠 Tech Stack

- [Ollama](https://ollama.ai) – LLM and embedding model runner  
- [ChromaDB](https://docs.trychroma.com) – Vector database  
- [Gradio](https://www.gradio.app) – User interface  
- [DuckDuckGo Search API](https://duckduckgo.com) – Web search fallback  
- [LangChain](https://python.langchain.com) – Pipeline orchestrator

---

## 💡 Example Use Cases

- 🤖 Customer support chatbot trained on your documentation  
- 🏢 Internal Q&A assistant for company handbooks or SOPs  
- 🛠 Technical manual and knowledge retrieval system  

---

## 🔗 References

- [Build a chatbot with RAG (Tom’s Tech Academy)](https://tomstechacademy.com/build-a-chatbot-with-rag-retrieval-augmented-generation/)
- [RAG: Retrieval-Augmented Generation Paper (Meta AI)](https://arxiv.org/abs/2005.11401)
- [Ollama Documentation](https://ollama.ai/docs)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [Gradio Documentation](https://www.gradio.app/guides)
- [DuckDuckGo Search API](https://duckduckgo.com)

