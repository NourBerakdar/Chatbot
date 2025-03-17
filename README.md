# RAG Chatbot with Ollama and Chroma

This is a Retrieval-Augmented Generation (RAG) chatbot built with Ollama, Chroma, and a fallback web search using DuckDuckGo. It ingests PDF documents into a vector store, retrieves relevant chunks to answer questions, and uses a Gradio interface for interaction. If no relevant information is found in the documents, it searches the web.

## Features
- Ingests PDF documents from a specified directory into a Chroma vector store.
- Uses Ollama for embeddings and language model generation.
- Streams responses in a Gradio UI.
- Falls back to DuckDuckGo web search when no relevant document chunks are found.

## Prerequisites
- Python 3.8+
- Ollama installed and running locally (see [Ollama docs](https://ollama.ai/))
- A directory of PDF files to ingest

## Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name


git remote add origin https://github.com/NourBerakdar/RAG_Chatbot.git
