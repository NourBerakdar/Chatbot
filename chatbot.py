"""A RAG chatbot using Ollama and Chroma with web search fallback."""

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from duckduckgo_search import DDGS
import gradio as gr
import config  # Import configuration
import time

def initialize_retriever():
    """Initialize the Chroma vector store and retriever."""
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PATH,
    )
    return vector_store.as_retriever(search_kwargs={"k": config.NUM_RESULTS})

def web_search(query, max_results=3):
    """Perform a web search using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Web search failed: {e}"

def generate_response(message, history, retriever, llm):
    """Generate a response based on retrieved documents or web search."""
    docs = retriever.invoke(message)
    knowledge = "\n\n".join([doc.page_content for doc in docs])

    if not knowledge.strip() or len(docs) == 0:
        knowledge = web_search(message)

    prompt = f"""
    You are an assistant that answers questions based on provided knowledge.
    Use only the information in the "The knowledge" section to answer, unless it's empty.
    Do not mention the source unless explicitly asked.

    The question: {message}
    Conversation history: {history}
    The knowledge: {knowledge}
    """

    partial_message = ""
    for chunk in llm.stream(prompt):
        partial_message += chunk
        yield partial_message  # Yield chunks for streaming

def main():
    """Launch the Gradio chatbot interface with streaming and processing time."""
    retriever = initialize_retriever()
    llm = OllamaLLM(model=config.LLM_MODEL)

    # Define the interface with a Chatbot and Textbox
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        textbox = gr.Textbox(placeholder="Ask me anything...", container=False, scale=7)

        def update_chat(message, history):
            # Initialize history if None
            if history is None:
                history = []
            # Append user message to history and yield immediately to show it
            history.append([message, ""])
            yield history, ""  # Show the user's message in the UI and clear the textbox
            
            # Measure processing time
            start_time = time.time()
            # Stream the assistant's response
            for partial_response in generate_response(message, history, retriever, llm):
                history[-1][1] = partial_response  # Update the assistant's response
                yield history, ""  # Continue yielding updated history and keep textbox clear
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Append processing time as a new assistant message, aligned to the right
            processing_message = f"<div style='text-align: right; font-style: italic; color: grey;'>Processed in {processing_time:.2f} seconds</div>"
            history.append([None, processing_message])
            yield history, ""  # Yield the final history with processing time

        # Submit event for streaming
        textbox.submit(
            update_chat, 
            inputs=[textbox, chatbot], 
            outputs=[chatbot, textbox]  # Update both chatbot and textbox
        )

    demo.launch()

if __name__ == "__main__":
    main()