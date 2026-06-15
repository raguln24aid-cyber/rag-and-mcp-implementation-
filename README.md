# 📚 RAG Chatbot with Web Search (Local Mistral + PDF)

A simple, fully local RAG (Retrieval-Augmented Generation) chatbot that answers questions from a PDF document (e.g., *Think and Grow Rich* by Napoleon Hill) and automatically falls back to live web search using DuckDuckGo when the PDF does not contain the answer.

## ✨ Features

- 🔍 **RAG on your PDF** – ask questions about the book and get answers from its content.
- 🌐 **Automatic web search** – the LLM decides when to search the internet (no hardcoded rules).
- 🧠 **Local LLM** – uses [Ollama](https://ollama.com/) with the Mistral model (no OpenAI API key required).
- 💾 **Persistent vector database** – indexes the PDF once and reuses it for future queries.
- 🐣 **No external APIs** – except free DuckDuckGo search.
- 🖥️ **Simple terminal chat** – interactive and easy to extend.

## 🧰 Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running
- The Mistral model pulled locally:  
  ```bash
  ollama pull mistral
