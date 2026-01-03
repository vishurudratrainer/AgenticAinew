import sys
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# --- Imports for SQL History ---
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_ollama import ChatOllama


# --- 1. Database Configuration ---
# !!! REPLACE THESE WITH YOUR ACTUAL MYSQL CREDENTIALS !!!
MYSQL_USER = "root"          # e.g., 'root'
MYSQL_PASSWORD = "root"  # e.g., 'secretpassword'
MYSQL_HOST = "localhost"
MYSQL_PORT = "3306"
MYSQL_DB = "langchain_db"               # Ensure this database exists

# The table name where history will be stored (LangChain manages this table)
HISTORY_TABLE = "ollama_chat_history"

# Construct the SQLAlchemy database URL
DB_URL = (
    f"mysql+mysqldb://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    A factory function that returns a SQLChatMessageHistory instance.
    """
    # 🔑 SQLChatMessageHistory uses the DB_URL and creates the table if it doesn't exist.
    # The session_id is used as a column value to filter messages for this specific conversation.
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=DB_URL,
        table_name=HISTORY_TABLE,
    )

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
            "You are a strict, efficient database assistant. All conversation facts must be recalled precisely."
        ),
        MessagesPlaceholder(variable_name="history"), 
        ("human", "{input}"),
    ]
)


# 3. Create the Chain with History

base_chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_session_history,
    input_messages_key="input", 
    history_messages_key="history",
)


# 4. Interactive Demonstration

if __name__ == "__main__":
    
    # Use a unique session ID for testing
    session_id = "mysql-session-789"

    print(f"\n{'='*70}")
    print(f"--- Starting MySQL-Backed Conversation (ID: {session_id}) ---")
    print(f"--- History will be saved/loaded from table: {HISTORY_TABLE}")
    print(f"{'='*70}")

    config = {"configurable": {"session_id": session_id}}

    # --- Turn 1: Provide a fact ---
    first_input = {"input": "I live in Berlin. What is the latest version of Python?"}
    print(f"\n[USER 1]: {first_input['input']}")
    response_1 = chain_with_history.invoke(first_input, config=config)
    print(f"[ASSISTANT 1]: {response_1.content}")
    
    print("\n--- INFO: Conversation history saved to MySQL. ---")

    # --- Turn 2: Ask a question that requires recalling the fact ---
    second_input = {"input": "What city do I live in?"}
    print(f"\n[USER 2]: {second_input['input']}")
    # The chain automatically loads the history from the MySQL table using the session_id.
    response_2 = chain_with_history.invoke(second_input, config=config)
    print(f"[ASSISTANT 2]: {response_2.content}")


    # --- History Check via the SQL object ---
    print("\n--- History Check (Verifying MySQL Content) ---")
    try:
        # 🔑 Create a new SQL history object instance to read the final state
        history_manager = get_session_history(session_id)
        history_messages = history_manager.messages
        
        print(f"Total messages stored in MySQL table: {len(history_messages)}")
        print(f"Last AI Response Content: {history_messages[-1].content}")
        
    except Exception as e:
        print(f"❌ Error accessing MySQL table or history: {e}")
        print("Please verify your MySQL credentials and that the database is running.")