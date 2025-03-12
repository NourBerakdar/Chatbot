from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
import gradio as gr
from dotenv import load_dotenv
import os

load_dotenv()

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

llm = InferenceClient(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    token=os.getenv("HUGGINGFACE_API_KEY")
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

num_results = 10
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

def stream_response(message, history):
    # Convert history to string for simplicity (optional: keep as list of dicts)
    history_str = "\n".join([f"{entry['role']}: {entry['content']}" for entry in history]) if history else ""

    docs = retriever.invoke(message)

    seen_content = set()
    knowledge = ""
    for doc in docs:
        content = doc.page_content.strip()
        if content and content not in seen_content:
            seen_content.add(content)
            knowledge += content + "\n\n"

    if not knowledge.strip():
        fallback_prompt = f"""
        I don’t have specific information from the documents to answer your question.
        I’ll provide a general response based on my training data instead.

        Question: {message}
        Conversation history: {history_str}
        """
        response = llm.text_generation(fallback_prompt, max_new_tokens=4096, temperature=0.5)
    else:
        rag_prompt = f"""
        You are an assistant that answers questions solely based on the provided knowledge.
        Do not use any external information or pre-trained knowledge beyond what is given below.
        If the knowledge contains a detailed section relevant to the question, return that section in full,
        preserving its structure (e.g., bullet points, headings) without summarizing or altering it.
        If the knowledge doesn’t fully answer the question, say so and stop there.

        Question: {message}
        Conversation history: {history_str}
        Knowledge: {knowledge}
        """
        response = llm.text_generation(rag_prompt, max_new_tokens=4096, temperature=0.5)

    # Stream the response as OpenAI-style message
    partial_message = ""
    for char in response:
        partial_message += char
        yield {"role": "assistant", "content": partial_message}

chatbot = gr.ChatInterface(
    stream_response,
    chatbot=gr.Chatbot(type="messages"),  # Explicitly set to messages format
    textbox=gr.Textbox(
        placeholder="Send to the LLM...",
        container=False,
        autoscroll=True,
        scale=7
    ),
)

chatbot.launch(share=True)