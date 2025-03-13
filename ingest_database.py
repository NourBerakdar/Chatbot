from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Initiate the embeddings model (using Ollama)
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")  # Replace with your preferred Ollama embedding model

# Initiate the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Load the PDF documents
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(raw_documents)

# Create unique IDs for chunks
uuids = [str(uuid4()) for _ in range(len(chunks))]

# Add chunks to vector store
vector_store.add_documents(documents=chunks, ids=uuids)

print(f"Successfully ingested {len(chunks)} chunks into the vector store.")