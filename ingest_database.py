"""Ingest PDF documents into a Chroma vector store using Ollama embeddings."""

import logging
from uuid import uuid4
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import config

logging.basicConfig(level=logging.INFO)

def initialize_vector_store() -> Chroma:
    """Initialize and return the Chroma vector store."""
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_PATH),
    )

def load_and_split_documents(data_path: str):
    """Load and split PDF documents into smaller chunks."""
    loader = PyPDFDirectoryLoader(data_path)
    raw_documents = loader.load()
    if not raw_documents:
        raise ValueError(f"No documents found in {data_path}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(raw_documents)

def ingest_data():
    """Ingest PDF chunks into the vector store."""
    try:
        vector_store = initialize_vector_store()
        chunks = load_and_split_documents(str(config.DATA_PATH))
        ids = [str(uuid4()) for _ in chunks]
        vector_store.add_documents(documents=chunks, ids=ids)
        logging.info(f"Successfully ingested {len(chunks)} chunks.")
    except Exception as e:
        logging.error(f"Error during ingestion: {e}", exc_info=True)

if __name__ == "__main__":
    ingest_data()