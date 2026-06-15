import os
import sys
from typing import List, Tuple
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings   # local embeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from duckduckgo_search import DDGS

# ------------------------------
# 1. Index the PDF (if not already)
# ------------------------------
PDF_PATH = "Think-And-Grow-Rich-Napoleon-Hill.pdf"
PERSIST_DIR = "./chroma_db"

def get_retriever():
    """Load or create the vector store retriever."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(PERSIST_DIR):
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        print("✅ Loaded existing vector database.")
    else:
        if not os.path.exists(PDF_PATH):
            print(f"❌ PDF not found: {PDF_PATH}")
            sys.exit(1)
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
        vectorstore.persist()
        print(f"✅ Indexed {len(chunks)} chunks from PDF.")
    
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# ------------------------------
# 2. Web search function
# ------------------------------
def web_search(query: str) -> str:
    """Search DuckDuckGo and return top 3 results as a string."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=30))
            if not results:
                return "No web results found."
            return "\n".join([f"[{i+1}] {r['body']}" for i, r in enumerate(results)])
    except Exception as e:
        return f"Web search failed: {str(e)}"

# ------------------------------
# 3. Query LLM with decision for web search
# ------------------------------
def answer_with_rag(query: str, retriever, llm) -> str:
    """
    Decide if web search is needed:
    - First retrieve relevant PDF chunks.
    - Ask LLM if the context answers the query. If not, request web search.
    - If web search is triggered, call it and then answer with both sources.
    """
    # Retrieve relevant chunks
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Prompt to decide if web search is required
    decision_prompt = f"""You are an AI assistant with access to a document (Think and Grow Rich book) and optionally the internet.
Your task: decide whether the query can be answered **fully** using only the provided document context.
If the query asks for **current information**, **recent events**, **weather**, **news**, or something **not present** in the context, you must request a web search.

Document context:
{context}

User query: {query}

Respond with EXACTLY one line:
- If NO web search is needed, start with "FINAL:" and then give the final answer (use the context).
- If web search IS needed, start with "SEARCH:" and then write the exact search query.

Examples:
FINAL: The book says that desire is the starting point of all achievement.
SEARCH: current stock market price of Tesla
"""

    decision = llm(decision_prompt).strip()
    
    if decision.startswith("FINAL:"):
        # Answer directly from PDF
        return decision[6:].strip()
    
    elif decision.startswith("SEARCH:"):
        # Extract search query, perform web search, then answer combining PDF + web
        search_query = decision[7:].strip()
        print(f"🌐 LLM decided to search the web for: {search_query}")
        web_results = web_search(search_query)
        
        # Final prompt with both sources
        final_prompt = f"""Use the following sources to answer the user's query.
If the document context conflicts with web results, prioritize web results for current facts.

Document context:
{context}

Web search results (DuckDuckGo):
{web_results}

User query: {query}

Answer concisely and helpfully:"""
        return llm(final_prompt).strip()
    
    else:
        # Fallback: retry with a simpler prompt
        fallback = llm(f"Answer this based on the document context if possible, otherwise say you need web search: {query}\nContext: {context}")
        return fallback.strip()

# ------------------------------
# 4. Main chat loop
# ------------------------------
def main():
    print("Loading retriever and LLM...")
    retriever = get_retriever()
    llm = Ollama(model="mistral", temperature=0.2)
    
    print("\n🤖 RAG Chatbot with Web Search (local Mistral + PDF)\n")
    print("Ask questions about 'Think and Grow Rich' or anything current.\nType 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        
        print("🤔 Thinking...")
        answer = answer_with_rag(user_input, retriever, llm)
        print(f"\n🤖 Bot: {answer}")

if __name__ == "__main__":
    main()