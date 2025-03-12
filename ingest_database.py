from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Keep larger chunks for context
    chunk_overlap=100,  # Enough overlap to maintain continuity
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(raw_documents)

# Minimal deduplication: only remove exact duplicates
seen_content = set()
unique_chunks = []
for chunk in chunks:
    content = chunk.page_content.strip()
    if content and content not in seen_content:
        seen_content.add(content)
        unique_chunks.append(chunk)

uuids = [str(uuid4()) for _ in range(len(unique_chunks))]
vector_store.add_documents(documents=unique_chunks, ids=uuids)

print(f"Added {len(unique_chunks)} unique chunks to the vector store.")