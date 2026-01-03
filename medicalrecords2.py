import os
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader # <-- NEW

# --- A. Data Loading from CSV ---
# In a real app, for PDF/Word/Excel, you would use loaders like 
# 'PyPDFLoader', 'UnstructuredExcelLoader', etc.

print("Loading documents from medical_data.csv...")
loader = CSVLoader(
    file_path="C:\ml\code\medical.csv",
    csv_args={
        'delimiter': ',',
        'quotechar': '"',
    }
)
documents = loader.load()

# --- B. Chunking and Embedding ---
# Each row of the CSV is now a LangChain Document. We still chunk for better retrieval.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 1. Initialize Ollama Embeddings (nomic-embed-text)
print("Initializing Ollama Embeddings and creating FAISS index...")
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. Create FAISS Vector Store
vectorstore = FAISS.from_documents(docs, ollama_embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 

# --- C. RAG Chain Definition ---
# 1. Initialize Ollama LLM (llama3)
ollama_llm = ChatOllama(model="llama3", temperature=0)

# 2. Define the RAG Prompt Template
RAG_PROMPT_TEMPLATE = """
You are a highly specialized medical assistant. Your task is to accurately and concisely answer the question
based ONLY on the medical records provided in the context below. Do not use external knowledge.
If the information is not in the context, state that explicitly.

CONTEXT:
{context}

QUESTION: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# 3. Construct the RAG Chain using LCEL
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | ollama_llm
    | StrOutputParser()
)

# --- D. Query the RAG System ---
user_query = "What condition does patient P1001 have related to joint pain?"

print(f"\n--- Querying CSV-Grounded RAG System ---")
print(f"User Query: {user_query}")
print("-" * 40)

# Execute the RAG chain. It will retrieve the relevant CSV row(s).
final_answer = rag_chain.invoke(user_query)

print(f"\n✅ LLM (Ollama) Answer:")
print(final_answer)

# Example 2: Querying the second patient
user_query_2 = "What were the vitals for Alice Smith (P1002) during her last checkup?"

print(f"\n--- Querying CSV-Grounded RAG System (Query 2) ---")
print(f"User Query: {user_query_2}")
print("-" * 40)

final_answer_2 = rag_chain.invoke(user_query_2)

print(f"\n✅ LLM (Ollama) Answer:")
print(final_answer_2)

"""
Key Takeaways for Data Loading
Document Loaders: The loader.load() step is the only part that changes when switching document types (e.g., from CSVLoader to PyPDFLoader). All loaders output a list of Document objects.

Metadata: When loading a CSV, the CSVLoader automatically includes the row details as metadata in the LangChain Document object, which helps the RAG system retrieve the source of the information.

Chunking: The RecursiveCharacterTextSplitter ensures that even if you load a large PDF or a massive CSV, the data is broken down into small, digestible chunks for accurate embedding and retrieval.
"""