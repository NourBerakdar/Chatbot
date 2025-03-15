"""Ingest PDF documents into a Chroma vector store using Ollama embeddings."""

from uuid import uuid4
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import config  # Import configuration

def initialize_vector_store():
    """Initialize the Chroma vector store with Ollama embeddings."""
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PATH,
    )

def load_and_split_documents(data_path):
    """Load PDF documents and split them into chunks."""
    loader = PyPDFDirectoryLoader(data_path)
    raw_documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(raw_documents)

def ingest_data():
    """Ingest documents into the vector store."""
    try:
        vector_store = initialize_vector_store()
        chunks = load_and_split_documents(config.DATA_PATH)
        uuids = [str(uuid4()) for _ in range(len(chunks))]
        vector_store.add_documents(documents=chunks, ids=uuids)
        print(f"Successfully ingested {len(chunks)} chunks into the vector store.")
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    ingest_data()