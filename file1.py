import os
import sys
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# --- CORRECTED IMPORTS ---
from langchain_ollama import ChatOllama
# 🔑 Import the File-Based History class
from langchain_community.chat_message_histories import FileChatMessageHistory 


# --- SETUP ---

# 1. Define the directory where history files will be saved
HISTORY_DIR = "chat_histories"

# Create the directory if it doesn't exist
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)
    print(f"--- INFO: Created history directory: {HISTORY_DIR}")


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    A factory function to create a FileChatMessageHistory instance for a session ID.
    The history will be saved to a specific JSON file in the HISTORY_DIR.
    """
    file_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    
    # 🔑 We return a new instance of FileChatMessageHistory, 
    # which automatically handles loading and saving to the file_path.
    return FileChatMessageHistory(file_path)

# Initialize the Ollama model.
try:
    llm = ChatOllama(model="mistral", temperature=0.0) 
    print("✅ Ollama model initialized with 'mistral'.")
except Exception as e:
    print(f"❌ Error initializing Ollama: {e}")
    print("Please ensure Ollama is running and the 'mistral' model is pulled.")
    sys.exit()


# 2. Define the Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful and friendly assistant. Keep your answers concise and directly address the user's question, remembering any facts they provide about themselves."
        ),
        # This is where history is injected
        MessagesPlaceholder(variable_name="history"), 
        ("human", "{input}"),
    ]
)


# 3. Create the Chain with History

base_chain = prompt | llm

# Wrap the chain with history management
chain_with_history = RunnableWithMessageHistory(
    runnable=base_chain,
    # This now calls our factory function which returns a FileChatMessageHistory instance
    get_session_history=get_session_history,
    input_messages_key="input", 
    history_messages_key="history",
)


## 4. Interactive Demonstration

if __name__ == "__main__":
    session_id = "user-file-session-001"
    history_file = os.path.join(HISTORY_DIR, f"{session_id}.json")

    print(f"\n{'='*70}")
    print(f"--- Starting File-Based Conversation (ID: {session_id}) ---")
    print(f"--- History will be saved/loaded from: {history_file}")
    print(f"{'='*70}")

    config = {"configurable": {"session_id": session_id}}

    # --- Turn 1: Providing a fact ---
    first_input = {"input": "My favorite programming language is Python. What is the capital of Canada?"}
    print(f"\n[USER 1]: {first_input['input']}")
    response_1 = chain_with_history.invoke(first_input, config=config)
    print(f"[ASSISTANT 1]: {response_1.content}")
    
    # At this point, the history file (user-file-session-001.json) has been created/updated.
    print(f"\n--- INFO: History saved to file. ---")


    # --- Turn 2: Asking a question that requires recalling the fact ---
    second_input = {"input": "Do you remember my favorite programming language?"}
    print(f"\n[USER 2]: {second_input['input']}")
    # The chain automatically loads the history from the file before invoking the LLM.
    response_2 = chain_with_history.invoke(second_input, config=config)
    print(f"[ASSISTANT 2]: {response_2.content}")


    # --- History Check from the file ---
    print("\n--- History Check (Verifying File Content) ---")
    try:
        # 🔑 Create a new history object to read the final state from the file
        final_history_manager = FileChatMessageHistory(history_file)
        history_messages = final_history_manager.messages
        
        print(f"Total messages stored in file: {len(history_messages)}")
        print(f"Last Human Message: {history_messages[-2].content}")
        print(f"Last AI Response: {history_messages[-1].content}")
        
    except Exception as e:
        print(f"❌ Error reading final history file: {e}")