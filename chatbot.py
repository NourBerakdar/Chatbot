"""A RAG chatbot using Ollama and Chroma with web search fallback."""

import logging
import time
import gradio as gr
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from duckduckgo_search import DDGS
import config

logging.basicConfig(level=logging.INFO)

def initialize_retriever():
    """Initialize Chroma retriever."""
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_PATH),
    )
    return vector_store.as_retriever(search_kwargs={"k": config.NUM_RESULTS})

def web_search(query: str, max_results: int = 3) -> str:
    """Perform fallback web search via DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        logging.warning(f"Web search failed: {e}")
        return "No relevant information found via web search."

def generate_response(message: str, history, retriever, llm):
    """Generate a response using retrieved documents or fallback search."""
    docs = retriever.invoke(message)
    knowledge = "\n\n".join([doc.page_content for doc in docs])

    if not knowledge.strip():
        logging.info("No relevant documents found. Falling back to web search.")
        knowledge = web_search(message)

    prompt = f"""You are a helpful assistant.
                Only answer based on the knowledge section below. If it's empty, you may generalize.

                Question: {message}
                Conversation history: {history}
                Knowledge: {knowledge}
                """

    partial_message = ""
    for chunk in llm.stream(prompt):
        partial_message += chunk
        yield partial_message  # Stream response

def main():
    """Launch chatbot UI with streaming."""
    retriever = initialize_retriever()
    llm = OllamaLLM(model=config.LLM_MODEL)

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        textbox = gr.Textbox(placeholder="Ask me anything...", container=False, scale=7)

        def update_chat(message, history):
            history = history or []
            history.append([message, "⏳"])  # Loading indicator
            yield history, ""
            
            start_time = time.time()
            for partial_response in generate_response(message, history, retriever, llm):
                history[-1][1] = partial_response
                yield history, ""
            end_time = time.time()

            processing_message = f"<div style='text-align: right; font-style: italic; color: grey;'>Processed in {end_time - start_time:.2f} sec</div>"
            history.append([None, processing_message])
            yield history, ""

        textbox.submit(
            update_chat, 
            inputs=[textbox, chatbot], 
            outputs=[chatbot, textbox]
        )

    demo.launch()

if __name__ == "__main__":
    main()
