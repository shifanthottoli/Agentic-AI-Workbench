# ============================================================
# AGENTIC AI CHATBOT WITH RAG + HITL
# ============================================================
#
# Features:
#
# - OpenRouter LLM
# - LangGraph
# - Tavily Search
# - Calculator
# - Stock Price
# - Weather
# - PDF RAG
# - FAISS Vector Database
# - SQLite Conversation Memory
# - LangSmith Thread Tracing
# - Human-in-the-Loop (HITL)
# - Purchase Stock Approval
#
# IMPORTANT:
#
# purchase_stock() is a MOCK tool.
# It does NOT connect to a real brokerage.
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import sqlite3
import math
import requests

from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.graph.message import add_messages

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.types import (
    interrupt,
    Command,
)

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_core.tools import tool

from langchain_openai import ChatOpenAI

from langchain_tavily import TavilySearch

from langchain_community.document_loaders import (
    PyPDFLoader,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

from langchain_community.vectorstores import (
    FAISS,
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 1. OPENROUTER CONFIGURATION
# ============================================================

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

if not OPENROUTER_API_KEY:

    raise ValueError(
        "OPENROUTER_API_KEY is not set. "
        "Add it to your .env file."
    )


OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free",
)


# ============================================================
# 2. TAVILY CONFIGURATION
# ============================================================

TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY"
)

if not TAVILY_API_KEY:

    raise ValueError(
        "TAVILY_API_KEY is not set. "
        "Add it to your .env file."
    )


# ============================================================
# 3. ALPHA VANTAGE CONFIGURATION
# ============================================================

ALPHA_VANTAGE_API_KEY = os.getenv(
    "ALPHA_VANTAGE_API_KEY"
)


# ============================================================
# 4. INITIALIZE OPENROUTER LLM
# ============================================================

llm = ChatOpenAI(

    model=OPENROUTER_MODEL,

    api_key=OPENROUTER_API_KEY,

    base_url="https://openrouter.ai/api/v1",

    temperature=0.7,
)


# ============================================================
# 5. EMBEDDING MODEL
# ============================================================

# ============================================================
# HUGGINGFACE EMBEDDING MODEL
# ============================================================

# IMPORTANT:
# The embedding model is loaded lazily.
# This prevents Streamlit from loading the model every time
# the application starts/reloads.

_embeddings = None


def get_embeddings():
    """
    Load HuggingFace embeddings only when RAG actually needs them.
    """

    global _embeddings

    if _embeddings is None:

        print(
            "Loading HuggingFace embedding model..."
        )

        _embeddings = HuggingFaceEmbeddings(

            model_name=(
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),

            model_kwargs={
                "device": "cpu"
            },

            encode_kwargs={
                "normalize_embeddings": True
            }
        )

        print(
            "HuggingFace embedding model loaded."
        )

    return _embeddings


# ============================================================
# 6. FAISS DATABASE PATH
# ============================================================

FAISS_DB_PATH = "faiss_db"


# ============================================================
# 7. GLOBAL VECTOR STORE
# ============================================================

vector_store = None


# ============================================================
# 8. LOAD EXISTING FAISS DATABASE
# ============================================================

def load_vector_store():

    global vector_store

    # --------------------------------------------------------
    # Already loaded
    # --------------------------------------------------------

    if vector_store is not None:

        return vector_store


    # --------------------------------------------------------
    # Check whether database exists
    # --------------------------------------------------------

    index_file = os.path.join(
        FAISS_DB_PATH,
        "index.faiss",
    )

    if not os.path.exists(index_file):

        return None


    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    try:

        vector_store = FAISS.load_local(

            folder_path=FAISS_DB_PATH,

            embeddings=embeddings,

            allow_dangerous_deserialization=True,
        )

        return vector_store


    except Exception as error:

        print(
            f"Could not load FAISS database: {error}"
        )

        return None


# ============================================================
# 9. INGEST PDF DOCUMENT
# ============================================================

def ingest_rag_document(
    file_path: str,
):

    global vector_store


    # --------------------------------------------------------
    # Load PDF
    # --------------------------------------------------------

    loader = PyPDFLoader(
        file_path
    )

    documents = loader.load()


    if not documents:

        raise ValueError(
            "The PDF does not contain readable text."
        )


    # --------------------------------------------------------
    # Split documents
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200,
    )


    chunks = splitter.split_documents(
        documents
    )


    if not chunks:

        raise ValueError(
            "No text chunks were created from the PDF."
        )


    # --------------------------------------------------------
    # Add filename metadata
    # --------------------------------------------------------

    filename = os.path.basename(
        file_path
    )


    for chunk in chunks:

        chunk.metadata["source"] = filename


    # --------------------------------------------------------
    # Create FAISS vector store
    # --------------------------------------------------------

    vector_store = FAISS.from_documents(

        documents=chunks,

        embedding=embeddings,
    )


    # --------------------------------------------------------
    # Save locally
    # --------------------------------------------------------

    os.makedirs(
        FAISS_DB_PATH,
        exist_ok=True,
    )


    vector_store.save_local(
        FAISS_DB_PATH
    )


    print(
        f"RAG document indexed successfully: {filename}"
    )

    print(
        f"Number of chunks: {len(chunks)}"
    )


    return {

        "filename": filename,

        "pages": len(documents),

        "chunks": len(chunks),
    }


# ============================================================
# 10. GET RAG RETRIEVER
# ============================================================

def get_retriever():

    global vector_store


    vector_store = load_vector_store()


    if vector_store is None:

        return None


    retriever = vector_store.as_retriever(

        search_type="similarity",

        search_kwargs={
            "k": 4,
        },
    )


    return retriever


# ============================================================
# 11. RAG TOOL
# ============================================================

@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the uploaded PDF.

    Use this tool when the user asks questions about
    an uploaded document.
    """

    retriever = get_retriever()


    # --------------------------------------------------------
    # No document
    # --------------------------------------------------------

    if retriever is None:

        return (
            "No document has been uploaded yet. "
            "Please upload a PDF document first."
        )


    # --------------------------------------------------------
    # Retrieve chunks
    # --------------------------------------------------------

    try:

        documents = retriever.invoke(
            query
        )


    except Exception as error:

        return (
            f"RAG retrieval error: {str(error)}"
        )


    # --------------------------------------------------------
    # No results
    # --------------------------------------------------------

    if not documents:

        return (
            "No relevant information was found "
            "in the uploaded document."
        )


    # --------------------------------------------------------
    # Format documents
    # --------------------------------------------------------

    formatted_documents = []


    for index, document in enumerate(
        documents,
        start=1,
    ):

        source = document.metadata.get(
            "source",
            "Unknown source",
        )


        page = document.metadata.get(
            "page",
            None,
        )


        if page is not None:

            page_display = page + 1

        else:

            page_display = "Unknown"


        formatted_documents.append(

            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page_display}\n"
            f"Content:\n"
            f"{document.page_content}"
        )


    return "\n\n".join(
        formatted_documents
    )


# ============================================================
# 12. TAVILY SEARCH TOOL
# ============================================================

search_tool = TavilySearch(

    max_results=5,

    topic="general",

    search_depth="advanced",
)


# ============================================================
# 13. CALCULATOR TOOL
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations.
    """

    try:

        allowed = {

            "math": math,

            "abs": abs,

            "round": round,

            "min": min,

            "max": max,

            "sum": sum,
        }


        result = eval(

            expression,

            {
                "__builtins__": {}
            },

            allowed,
        )


        return str(result)


    except Exception as error:

        return (
            f"Calculation error: {str(error)}"
        )


# ============================================================
# 14. STOCK PRICE TOOL
# ============================================================

@tool
def get_stock_price(
    symbol: str,
) -> dict:
    """
    Get the latest stock price.
    """

    if not ALPHA_VANTAGE_API_KEY:

        return {

            "error":
            "ALPHA_VANTAGE_API_KEY is not "
            "configured in .env",
        }


    symbol = symbol.upper().strip()


    url = (

        "https://www.alphavantage.co/query"

        "?function=GLOBAL_QUOTE"

        f"&symbol={symbol}"

        f"&apikey={ALPHA_VANTAGE_API_KEY}"
    )


    try:

        response = requests.get(

            url,

            timeout=10,
        )


        response.raise_for_status()


        data = response.json()


        if "Global Quote" not in data:

            return {

                "error":
                f"No stock data found for {symbol}.",
            }


        quote = data[
            "Global Quote"
        ]


        if not quote:

            return {

                "error":
                f"No stock data found for {symbol}.",
            }


        return {

            "symbol":
            quote.get(
                "01. symbol"
            ),

            "price":
            quote.get(
                "05. price"
            ),

            "change":
            quote.get(
                "09. change"
            ),

            "change_percent":
            quote.get(
                "10. change percent"
            ),

            "volume":
            quote.get(
                "06. volume"
            ),
        }


    except Exception as error:

        return {

            "error":
            f"Stock API error: {str(error)}",
        }


# ============================================================
# 15. WEATHER TOOL
# ============================================================

@tool
def get_current_weather(
    city: str,
) -> dict:
    """
    Get current weather for a city.
    """

    city = city.strip()


    url = (

        f"https://wttr.in/"
        f"{city}?format=j1"
    )


    try:

        response = requests.get(

            url,

            timeout=10,
        )


        response.raise_for_status()


        data = response.json()


        current = data[
            "current_condition"
        ][0]


        return {

            "city": city,

            "temperature_C":
            current.get(
                "temp_C"
            ),

            "feels_like_C":
            current.get(
                "FeelsLikeC"
            ),

            "humidity":
            current.get(
                "humidity"
            ),

            "weather":
            current[
                "weatherDesc"
            ][0].get(
                "value"
            ),

            "wind_speed_kmph":
            current.get(
                "windspeedKmph"
            ),
        }


    except Exception as error:

        return {

            "error":
            f"Weather API error: {str(error)}",
        }


# ============================================================
# 16. HUMAN-IN-THE-LOOP STOCK PURCHASE TOOL
# ============================================================

@tool
def purchase_stock(
    symbol: str,
    quantity: int,
) -> dict:
    """
    Simulate purchasing a given quantity of a stock.

    IMPORTANT:
    This is a MOCK purchase tool.

    It does NOT connect to a real brokerage.

    Before completing the purchase, the tool pauses
    execution and asks a human for approval.

    The human must respond with:
        yes
    or:
        anything else = reject
    """

    # --------------------------------------------------------
    # Validate symbol
    # --------------------------------------------------------

    symbol = symbol.upper().strip()


    # --------------------------------------------------------
    # Validate quantity
    # --------------------------------------------------------

    try:

        quantity = int(quantity)

    except Exception:

        return {

            "status": "error",

            "message":
            "Quantity must be an integer.",
        }


    if quantity <= 0:

        return {

            "status": "error",

            "message":
            "Quantity must be greater than zero.",
        }


    # ========================================================
    # HUMAN-IN-THE-LOOP INTERRUPTION
    # ========================================================
    #
    # The graph pauses here.
    #
    # The Streamlit application detects this interrupt
    # and shows Approve / Reject buttons.
    #
    # When the human responds, LangGraph resumes here.
    #
    # ========================================================

    decision = interrupt(

        {

            "type": "purchase_approval",

            "action": "purchase_stock",

            "symbol": symbol,

            "quantity": quantity,

            "message":
            (
                f"Approve buying {quantity} "
                f"shares of {symbol}?"
            ),
        }
    )


    # ========================================================
    # PROCESS HUMAN DECISION
    # ========================================================

    if (

        isinstance(
            decision,
            str,
        )

        and

        decision.lower().strip()
        == "yes"
    ):

        return {

            "status": "success",

            "message":
            (
                f"Purchase order placed for "
                f"{quantity} shares of {symbol}."
            ),

            "symbol": symbol,

            "quantity": quantity,
        }


    # --------------------------------------------------------
    # Human rejected
    # --------------------------------------------------------

    return {

        "status": "cancelled",

        "message":
        (
            f"Purchase of {quantity} shares "
            f"of {symbol} was declined by human."
        ),

        "symbol": symbol,

        "quantity": quantity,
    }


# ============================================================
# 17. TOOL LIST
# ============================================================

tools = [

    # --------------------------------------------------------
    # Web search
    # --------------------------------------------------------

    search_tool,


    # --------------------------------------------------------
    # Calculator
    # --------------------------------------------------------

    calculator,


    # --------------------------------------------------------
    # Stock price
    # --------------------------------------------------------

    get_stock_price,


    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    get_current_weather,


    # --------------------------------------------------------
    # PDF RAG
    # --------------------------------------------------------

    rag_tool,


    # --------------------------------------------------------
    # HUMAN-IN-THE-LOOP PURCHASE
    # --------------------------------------------------------

    purchase_stock,
]


# ============================================================
# 18. MAKE LLM TOOL-AWARE
# ============================================================

llm_with_tools = llm.bind_tools(
    tools
)


# ============================================================
# 19. CHAT STATE
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]


# ============================================================
# 20. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """

You are a helpful Agentic AI assistant.

You have access to the following tools:

1. rag_tool
   Use this for questions about an uploaded PDF
   or document.

2. Tavily search
   Use this for current information, recent events,
   news, internet searches, or information requiring
   web access.

3. calculator
   Use this for mathematical calculations.

4. get_stock_price
   Use this when the user asks for current stock
   prices or stock market information.

5. get_current_weather
   Use this when the user asks for current weather.

6. purchase_stock
   Use this when the user explicitly asks to purchase,
   buy, or place an order for stock.

IMPORTANT PURCHASE RULE:

- purchase_stock is a human-in-the-loop tool.
- NEVER pretend that a stock purchase was completed
  without calling purchase_stock.
- purchase_stock will pause and wait for human approval.
- The human must approve before the simulated purchase
  can be completed.
- If the human rejects the purchase, clearly tell the
  user that the purchase was cancelled.
- This purchase tool is only a MOCK implementation.
- It does NOT connect to a real brokerage.

IMPORTANT RAG RULES:

- If the user asks about the uploaded document,
  use rag_tool.

- Do not make up information from the document.

- Base document-related answers on retrieved
  document content.

- If no document is available, tell the user to
  upload a document.

- If the answer cannot be found in the uploaded
  document, clearly say that the information was
  not found in the uploaded document.

IMPORTANT TOOL RULES:

- Use calculator for mathematical calculations.
- Use Tavily for current or web-based information.
- Use stock tool for stock prices.
- Use weather tool for current weather.
- Use purchase_stock for stock purchase requests.
- Use rag_tool for document questions.

For general questions that do not require a tool,
answer directly.

After receiving a tool result, provide a clear,
natural-language final answer.
"""


# ============================================================
# 21. CHATBOT NODE
# ============================================================

def chat_node(
    state: ChatState,
):

    messages = [

        SystemMessage(
            content=SYSTEM_PROMPT
        )

    ] + state["messages"]


    response = llm_with_tools.invoke(
        messages
    )


    return {

        "messages": [
            response
        ]
    }


# ============================================================
# 22. CREATE STATE GRAPH
# ============================================================

graph = StateGraph(
    ChatState
)


# ============================================================
# 23. CHATBOT NODE
# ============================================================

graph.add_node(
    "chat_node",
    chat_node,
)


# ============================================================
# 24. TOOL NODE
# ============================================================

graph.add_node(
    "tools",
    ToolNode(tools),
)


# ============================================================
# 25. START → CHATBOT
# ============================================================

graph.add_edge(
    START,
    "chat_node",
)


# ============================================================
# 26. CHATBOT → TOOL OR END
# ============================================================

graph.add_conditional_edges(

    "chat_node",

    tools_condition,
)


# ============================================================
# 27. TOOL → CHATBOT
# ============================================================

graph.add_edge(
    "tools",
    "chat_node",
)


# ============================================================
# 28. SQLITE DATABASE
# ============================================================

conn = sqlite3.connect(

    database="chatbot.db",

    check_same_thread=False,
)


# ============================================================
# 29. SQLITE CHECKPOINTER
# ============================================================

checkpoint = SqliteSaver(
    conn
)


# ============================================================
# 30. COMPILE GRAPH
# ============================================================

chatbot = graph.compile(

    checkpointer=checkpoint,
)


# ============================================================
# 31. GET ALL THREADS
# ============================================================

def get_all_threads():

    all_threads = set()


    for ckpt in checkpoint.list(None):

        try:

            thread_id = ckpt.config[
                "configurable"
            ][
                "thread_id"
            ]


            all_threads.add(
                str(thread_id)
            )


        except (
            KeyError,
            TypeError,
        ):

            continue


    return list(
        all_threads
    )


# ============================================================
# 32. GET THREAD HISTORY
# ============================================================

def get_thread_history(
    thread_id: str,
):

    config = {

        "configurable": {

            "thread_id":
            str(thread_id),
        }
    }


    state = chatbot.get_state(
        config
    )


    if (

        state.values

        and

        "messages"
        in state.values
    ):

        return state.values[
            "messages"
        ]


    return []


# ============================================================
# 33. GET PENDING INTERRUPT
# ============================================================

def get_pending_interrupt(
    thread_id: str,
):
    """
    Check whether a LangGraph thread is currently
    waiting for human input.
    """

    config = {

        "configurable": {

            "thread_id":
            str(thread_id),
        }
    }


    try:

        state = chatbot.get_state(
            config
        )


        # ----------------------------------------------------
        # LangGraph stores pending interrupts in tasks.
        # ----------------------------------------------------

        if not state:

            return None


        tasks = getattr(
            state,
            "tasks",
            (),
        )


        for task in tasks:

            interrupts = getattr(
                task,
                "interrupts",
                (),
            )


            if interrupts:

                interrupt_value = (
                    interrupts[0].value
                )

                return interrupt_value


    except Exception as error:

        print(
            f"Could not read interrupt: {error}"
        )


    return None


# ============================================================
# 34. RESUME INTERRUPTED GRAPH
# ============================================================

def resume_chat(
    thread_id: str,
    decision: str,
):
    """
    Resume a paused LangGraph execution.

    decision should normally be:
        yes
        no
    """

    config = {

        "configurable": {

            "thread_id":
            str(thread_id),
        },

        "metadata": {

            "thread_id":
            str(thread_id),
        },

        "run_name":
        "chat_trace",
    }


    result = chatbot.invoke(

        Command(
            resume=decision
        ),

        config=config,
    )


    return result


# ============================================================
# 35. CHAT HELPER
# ============================================================

def chat(
    user_message: str,
    thread_id: str = "1",
):

    thread_id = str(
        thread_id
    )


    config = {

        "configurable": {

            "thread_id":
            thread_id,
        },

        "metadata": {

            "thread_id":
            thread_id,
        },

        "run_name":
        "chat_trace",
    }


    result = chatbot.invoke(

        {

            "messages": [

                HumanMessage(
                    content=user_message
                )
            ]
        },

        config=config,
    )


    messages = result.get(
        "messages",
        [],
    )


    if messages:

        return messages[-1].content


    return ""


# ============================================================
# 36. TOOL ALIASES
# ============================================================

tavily_search = search_tool

stock_price = get_stock_price

weather = get_current_weather


# ============================================================
# 37. TERMINAL HITL TEST
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 70)

    print(
        "       AGENTIC AI CHATBOT + RAG + HITL"
    )

    print("=" * 70)

    print(
        f"Model       : {OPENROUTER_MODEL}"
    )

    print(
        "Memory      : SQLite"
    )

    print(
        "RAG         : FAISS"
    )

    print(
        "Embeddings  : HuggingFace"
    )

    print(
        "Web Search  : Tavily"
    )

    print(
        "Calculator  : Enabled"
    )

    print(
        "Stock       : Alpha Vantage"
    )

    print(
        "Weather     : wttr.in"
    )

    print(
        "HITL        : Enabled"
    )

    print(
        "Purchase    : MOCK"
    )

    print("=" * 70)

    print(
        "Type 'exit' to stop."
    )

    print("=" * 70)


    thread_id = "terminal-hitl"


    while True:

        user_message = input(
            "\nYou: "
        ).strip()


        if user_message.lower() in [
            "exit",
            "quit",
            "bye",
        ]:

            print(
                "\nChat ended."
            )

            break


        if not user_message:

            continue


        config = {

            "configurable": {

                "thread_id":
                thread_id,
            },

            "metadata": {

                "thread_id":
                thread_id,
            },

            "run_name":
            "chat_trace",
        }


        try:

            result = chatbot.invoke(

                {

                    "messages": [

                        HumanMessage(
                            content=user_message
                        )
                    ]
                },

                config=config,
            )


            # ------------------------------------------------
            # Check for HITL interruption
            # ------------------------------------------------

            pending = get_pending_interrupt(
                thread_id
            )


            if pending:

                print()

                print(
                    "⚠️ HUMAN APPROVAL REQUIRED"
                )

                print("-" * 50)

                print(
                    pending.get(
                        "message",
                        "Approval required.",
                    )
                )

                print()

                decision = input(
                    "Approve? (yes/no): "
                ).strip().lower()


                if decision == "yes":

                    resume_result = resume_chat(

                        thread_id,

                        "yes",
                    )

                else:

                    resume_result = resume_chat(

                        thread_id,

                        "no",
                    )


                messages = (
                    resume_result.get(
                        "messages",
                        [],
                    )
                )


                if messages:

                    print(
                        "\nAI:",
                        messages[-1].content,
                    )

            else:

                messages = result.get(
                    "messages",
                    [],
                )


                if messages:

                    print(
                        "\nAI:",
                        messages[-1].content,
                    )


        except Exception as error:

            print(
                "\nError:",
                error,
            )