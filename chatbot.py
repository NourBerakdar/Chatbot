# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from duckduckgo_search import DDGS  # For web search fallback
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Initiate the embeddings model (Ollama)
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")  # Replace with your preferred Ollama embedding model

# Initiate the Ollama language model
llm = Ollama(model="llama3")  # Replace with your preferred Ollama model, e.g., "mistral", "llama3", etc.

# Connect to the Chroma vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Set up the retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# Web search function (fallback)
def web_search(query, max_results=3):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        web_knowledge = "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    return web_knowledge

# Stream response function
def stream_response(message, history):
    # Retrieve relevant chunks from the vector store
    docs = retriever.invoke(message)
    knowledge = "\n\n".join([doc.page_content for doc in docs])

    # Check if there's sufficient knowledge from files
    if not knowledge.strip() or len(docs) == 0:
        # Fallback to web search
        knowledge = web_search(message)
        source = "the internet"
    else:
        source = "provided files"

    # Construct the prompt
    rag_prompt = f"""
    You are an assistant that answers questions based on the provided knowledge.
    Use only the information in the "The knowledge" section to answer, unless it's empty.
    Do not mention the source of the knowledge to the user unless explicitly asked.

    The question: {message}

    Conversation history: {history}

    The knowledge: {knowledge}
    """

    # Stream the response
    partial_message = ""
    for response in llm.stream(rag_prompt):
        partial_message += response
        yield partial_message

# Initiate the Gradio app
chatbot = gr.ChatInterface(
    stream_response,
    textbox=gr.Textbox(
        placeholder="Ask me anything...",
        container=False,
        autoscroll=True,
        scale=7
    ),
)

# Launch the Gradio app
chatbot.launch()