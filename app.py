# app.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from duckduckgo_search import DDGS

PDF_PATH = "Think-And-Grow-Rich-Napoleon-Hill.pdf"
PERSIST_DIR = "./chroma_db"

# Initialize retriever (indexes PDF once)
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(PERSIST_DIR):
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"Error: {PDF_PATH} not found. Please place the PDF in the project folder.")
        loader = PyPDFLoader(PDF_PATH)
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# Web search
def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            return "\n".join([f"- {r['body']}" for r in results]) if results else "No results."
    except Exception as e:
        return f"Web search error: {e}"

# Main answer function (RAG + optional web search)
def answer_with_rag(query: str, retriever, llm) -> str:
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    decision_prompt = f"""Context: {context}
User: {query}
If context answers fully, output "FINAL: answer". If web search needed, output "SEARCH: exact query".
Answer only with "FINAL:" or "SEARCH:"."""
    decision = llm.invoke(decision_prompt).strip()
    if decision.startswith("FINAL:"):
        return decision[6:].strip()
    elif decision.startswith("SEARCH:"):
        search_query = decision[7:].strip()
        web_results = web_search(search_query)
        final_prompt = f"Context: {context}\nWeb search results: {web_results}\nUser: {query}\nAnswer using both sources:"
        return llm.invoke(final_prompt).strip()
    else:
        return llm.invoke(f"Answer using context: {query}\nContext: {context}").strip()