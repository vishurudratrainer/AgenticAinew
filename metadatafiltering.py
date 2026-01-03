"""
Docstring for agentloop.metadatafiltering
Metadata Filtering (Pre-Filtering)
When you have different categories of data
 (e.g., patient notes vs. lab results), 
 you can use metadata attached to the documents to narrow the search before the LLM runs. This makes the search faster and more accurate.
"""
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- A. Documents with Metadata ---
# Metadata allows us to filter the documents before they are retrieved.
trial_docs = [
    Document(page_content="Trial ID T001: Investigating drug X for cancer. Side effects were minimal. Efficacy results pending.", 
             metadata={"phase": "Phase 2", "drug": "Drug X"}),
    Document(page_content="Trial ID T002: Safety assessment of Drug Y in healthy volunteers. Completed with no serious adverse events.", 
             metadata={"phase": "Phase 1", "drug": "Drug Y"}),
    Document(page_content="Trial ID T003: Large-scale efficacy study of Drug Z. Shows significant improvement in patient outcomes.", 
             metadata={"phase": "Phase 3", "drug": "Drug Z"}),
]

# --- B. Embedding and Filtering Setup ---
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(trial_docs, ollama_embeddings)

# Define a **specific retriever** that only retrieves documents where 'phase' equals 'Phase 1'
phase_1_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {"phase": "Phase 1"} # This is the key filtering step
    }
)

# --- C. RAG Chain and Query ---
ollama_llm = ChatOllama(model="llama3", temperature=0)
rag_prompt = ChatPromptTemplate.from_template("Answer the question based ONLY on the context: {context}\n\nQuestion: {question}")

# Chain uses the pre-filtered retriever
rag_chain_filtered = (
    {"context": phase_1_retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | ollama_llm
    | StrOutputParser()
)

user_query = "Summarize the findings of the Phase 1 trial regarding Drug Y."

print(f"\n--- Metadata Filtered RAG System (Phase 1 Trials Only) ---")
print(f"User Query: {user_query}")
print("-" * 40)

# The retriever will ignore T001 (Phase 2) and T003 (Phase 3) regardless of semantic similarity.
final_answer = rag_chain_filtered.invoke(user_query)

print(f"\n✅ LLM (Ollama) Answer:")
print(final_answer)