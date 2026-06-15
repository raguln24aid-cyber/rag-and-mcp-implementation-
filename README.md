# 📚 RAG Chatbot with Web Search (Local Mistral + PDF)

This project provides a fully local RAG (Retrieval-Augmented Generation) chatbot. It's designed to answer questions using a provided PDF document (like *Think and Grow Rich* by Napoleon Hill) and intelligently perform live web searches via DuckDuckGo when the PDF content doesn't suffice.

## ✨ Features

- 🔍 **RAG on your PDF** – ask questions about the book and get answers from its content.
- 🌐 **Automatic web search** – the LLM decides when to search the internet (no hardcoded rules).
- 🧠 **Local LLM** – uses [Ollama](https://ollama.com/) with the Mistral model (no OpenAI API key required).
- 💾 **Persistent vector database** – the PDF is indexed once into a Chroma vector store, allowing for quick retrieval in subsequent sessions.
- 🐣 **No external APIs** – except free DuckDuckGo search.
- 🖥️ **Simple terminal chat** – interactive and easy to extend.

## 🚀 Getting Started

Follow these steps to set up and run the RAG Chatbot locally.

### 🧰 Requirements

*   **Python 3.9+**: Ensure you have a compatible Python version installed.
*   **Ollama**:
    1.  Download and install [Ollama](https://ollama.com/) for your operating system.
    2.  Pull the Mistral model by running the following command in your terminal:
        ```bash
        ollama pull mistral
        ```
    3.  Make sure Ollama is running in the background before starting the chatbot.

### 📦 Installation

1.  Clone this repository or download the project files.
2.  Navigate to the project directory in your terminal.
3.  Install the required Python libraries:
    ```bash
    pip install langchain langchain-community langchain-chroma sentence-transformers duckduckgo-search gradio pypdf
    ```
4.  Place your PDF file named `Think-And-Grow-Rich-Napoleon-Hill.pdf` in the root of the project directory.

### ▶️ How to Run

1.  Ensure all requirements are met and dependencies are installed.
2.  From the project's root directory, run the Gradio user interface:
    ```bash
    python ui.py
    ```
3.  A local URL (e.g., `http://127.0.0.1:7860`) will be displayed in your terminal. Open this URL in your web browser to interact with the chatbot.
