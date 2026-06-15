# ui.py
import gradio as gr
from app import get_retriever, answer_with_rag
from langchain_community.llms import Ollama

# Load models once
print("Loading retriever and LLM...")
retriever = get_retriever()
llm = Ollama(model="mistral", temperature=0.2)

# Chat function for Gradio
def chat_fn(message, history):
    if not message:
        return ""
    return answer_with_rag(message, retriever, llm)

# Create the interface
demo = gr.ChatInterface(
    fn=chat_fn,
    title="📚 RAG Chatbot (Think & Grow Rich + Web Search)",
    description="Ask about the book or any current topic. The bot will search the web if needed.",
)

if __name__ == "__main__":
    demo.launch()