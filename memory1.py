from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, Any

## 🛠️ Configuration and History Setup

# Type-hint the store for clarity
store: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """A factory function to retrieve or create a chat history for a session."""
    if session_id not in store:
        # 🔑 Crucial step: Instantiate the concrete history class
        store[session_id] = InMemoryChatMessageHistory() 
        print(f"--- INFO: Created new session history for ID: {session_id}")
    return store[session_id]

# Initialize the Ollama model.
# NOTE: Using 'mistral' as requested. Ensure it's pulled via 'ollama pull mistral'.
try:
    llm = ChatOllama(model="mistral", temperature=0.0) 
    print("✅ Ollama model initialized with 'mistral'.")
except Exception as e:
    print(f"❌ Error initializing Ollama: {e}")
    print("Please ensure Ollama is running and the 'mistral' model is pulled.")
    # Exit gracefully if LLM setup fails
    exit()

# ----------------------------------------------------

## 2. Define the Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful and friendly assistant. Keep your answers concise and directly address the user's question, remembering any facts they provide about themselves."
        ),
        # This placeholder is crucial; LangChain injects the history here.
        MessagesPlaceholder(variable_name="history"), 
        ("human", "{input}"),
    ]
)

# ----------------------------------------------------

## 3. Create the Chain with History

# 1. Define the base chain (Prompt + LLM)
base_chain = prompt | llm

# 2. Wrap the chain with history management
# 
chain_with_history = RunnableWithMessageHistory(
    runnable=base_chain,
    # This tells the system how to get the history object
    get_session_history=get_session_history,
    # Configurable keys
    input_messages_key="input", 
    history_messages_key="history",
)

# ----------------------------------------------------

## 4. Interactive Demonstration

session_id = "user-demo-123"
print(f"\n--- Starting Conversation (Session ID: {session_id}) ---")

# The configuration dict tells RunnableWithMessageHistory which session to use
config = {"configurable": {"session_id": session_id}}


# --- Turn 1: Providing a fact and asking a general question ---
first_input = {"input": "My cat's name is Mittens. What is the tallest mountain on Earth?"}
print(f"\n[USER 1]: {first_input['input']}")
response_1 = chain_with_history.invoke(first_input, config=config)
print(f"[ASSISTANT 1]: {response_1.content}")


# --- Turn 2: Asking a question that requires recalling the fact ---
second_input = {"input": "What is the name of my cat?"}
print(f"\n[USER 2]: {second_input['input']}")
response_2 = chain_with_history.invoke(second_input, config=config)
print(f"[ASSISTANT 2]: {response_2.content}")


# --- Turn 3: Asking a follow-up about the general question ---
third_input = {"input": "What country is it located in?"}
print(f"\n[USER 3]: {third_input['input']}")
response_3 = chain_with_history.invoke(third_input, config=config)
print(f"[ASSISTANT 3]: {response_3.content}")


# --- History Check ---
print("\n--- History Check ---")
try:
    # 🔑 This is the line that failed: accessing the history from the global store
    history_messages = store[session_id].messages
    
    # Successful access indicates the object is a valid ChatMessageHistory instance
    print(f"Total messages in history: {len(history_messages)}")
    print(f"Last stored message (AI): {history_messages[-1].content[:50]}...")
    
except KeyError:
    print(f"❌ Error: Session ID '{session_id}' not found in store.")
except Exception as e:
    print(f"❌ Unexpected Error during history retrieval: {e}")

# ----------------------------------------------------
