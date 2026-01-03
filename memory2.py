import sys
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# --- CORRECTED IMPORTS BASED ON YOUR ENVIRONMENT ---
# 1. Ollama is in the community package
from langchain_ollama import ChatOllama
# 2. The core in-memory class is imported this way in recent versions
from langchain_community.chat_message_histories import ChatMessageHistory


## 🛠️ Configuration and History Setup

# A dictionary to store history for different sessions (mimics a database)
store: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """A factory function to retrieve or create a chat history for a session."""
    if session_id not in store:
        # Using the CORRECTED class name found in your package structure
        store[session_id] = ChatMessageHistory() 
        print(f"--- INFO: Created new session history for ID: {session_id}")
    return store[session_id]

# Initialize the Ollama model.
try:
    # Using 'mistral' as you specified earlier
    llm = ChatOllama(model="mistral", temperature=0.0) 
    print("✅ Ollama model initialized with 'mistral'.")
except Exception as e:
    print(f"❌ Error initializing Ollama: {e}")
    print("Please ensure Ollama is running and the 'mistral' model is pulled.")
    sys.exit()


# 2. Define the Prompt Template

# The MessagesPlaceholder is crucial; LangChain automatically injects the history here.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful and friendly assistant. Keep your answers concise and directly address the user's question, remembering any facts they provide about themselves."
        ),
        # 🔑 This is where history goes
        MessagesPlaceholder(variable_name="history"), 
        ("human", "{input}"),
    ]
)


# 3. Create the Chain with History

# 1. Define the base chain using LCEL (Prompt | LLM)
base_chain = prompt | llm


# 2. Wrap the chain with history management
chain_with_history = RunnableWithMessageHistory(
    runnable=base_chain,
    # This tells the system how to get the history object
    get_session_history=get_session_history,
    # Configurable keys, telling the chain where to find the input and where to put history
    input_messages_key="input", 
    history_messages_key="history",
)


## 4. Interactive Demonstration

if __name__ == "__main__":
    session_id = "user-demo-123"
    print(f"\n{'='*50}")
    print(f"--- Starting Conversation (Session ID: {session_id}) ---")
    print(f"{'='*50}")

    # The configuration MUST include the session_id so the chain knows which history to use.
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

    # ----------------------------------------------------
    print("\n--- History Check ---")
    try:
        # 🔑 FINAL CORRECT LINE: Access the history using the .messages attribute
        history_messages = store[session_id].messages
        
        print(f"Total messages in history: {len(history_messages)}")
        # Example of the final message stored:
        print(f"Last stored message (AI): {history_messages[-1].content}")
        
    except Exception as e:
        print(f"❌ Error during final history retrieval: {e}")