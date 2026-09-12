# Agentic-AI-Workbench

Agentic-AI-Workbench is a full-stack prototype for an agentic AI assistant built with Streamlit, LangGraph, OpenRouter-backed LLMs, a local FAISS vector database, and a human-in-the-loop stock purchase approval workflow.

The application gives the user a chat-like workspace where they can:

- Ask questions to an LLM with tool access.
- Search the live web through Tavily.
- Perform calculations with a calculator tool.
- Ask for current stock prices using Alpha Vantage.
- Ask for weather information using a public weather endpoint.
- Upload a PDF and query it through a retrieval-augmented generation workflow.
- Trigger a simulated stock purchase, which pauses and requires approval before continuing.

This repository combines a Streamlit UI in the frontend file and the LangGraph orchestration, tool registry, vector store, and checkpointing logic in the backend file.

## Project Structure

- `app.py` — Streamlit UI, conversation rendering, uploaded PDF handling, chat input, and human approval buttons.
- `backend.py` — LangGraph chatbot graph, tools, RAG ingestion, FAISS retrieval, SQLite checkpointing, and interrupt/resume logic.
- `requirements.txt` — Python dependency list.
- `Dockerfile` — container image for running the Streamlit app.

## Main Features

### 1. LangGraph Agent Workflow

The backend creates a `StateGraph` using LangGraph and compiles a chatbot graph with a `chat_node`, a `ToolNode`, and conditional routing through `tools_condition`. The graph supports thread-based conversation memory through a SQLite checkpoint.

### 2. Tool-Calling Assistant

The LLM is made tool-aware by binding the tools list to `llm.bind_tools(tools)` in the backend. The available tools include:

- `search_tool` using Tavily.
- `calculator` for equations.
- `get_stock_price` using Alpha Vantage.
- `get_current_weather` using `wttr.in`.
- `rag_tool` for retrieving content from an uploaded PDF.
- `purchase_stock` for a human-in-the-loop mock stock purchase request.

### 3. PDF RAG

The repository implements a document ingestion pipeline that loads a PDF through `PyPDFLoader`, splits it with a `RecursiveCharacterTextSplitter`, embeds the chunks using HuggingFace embeddings, and stores the result in a local FAISS directory named `faiss_db`.

The RAG tool retrieves the top relevant chunks and returns them as structured context for the LLM.

### 4. Human-in-the-Loop Approval

The `purchase_stock` tool is designed as a mock workflow. It pauses execution with `interrupt(...)` and sends an approval payload containing the purchase request details. The Streamlit UI detects the pending interrupt and shows Approve / Reject controls. Once the user approves or rejects the action, the graph resumes through `resume_chat()`.

> Important: the purchase operation is simulated and does not connect to a real brokerage or financial market system.

### 5. Threaded Conversation History

The application tracks each conversation using a `thread_id` and persists conversation messages through the LangGraph SQLite checkpointer. The UI isolates the current chat context and allows the user to open old chat threads from the sidebar.

## Runtime Setup

The project expects the following environment variables in a `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=openrouter/free
TAVILY_API_KEY=your_tavily_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

The `OPENROUTER_API_KEY` and `TAVILY_API_KEY` are mandatory for the app to run. `ALPHA_VANTAGE_API_KEY` is needed for the stock-price tool.

## Local Installation

1. Create and activate a virtual environment.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the required API keys.
4. Start the Streamlit app:

```bash
streamlit run app.py
```

## Docker

A `Dockerfile` is included for containerized deployment. It installs Python dependencies, copies the workspace, exposes port `8501`, and starts the app with the Streamlit command.

Example:

```bash
docker build -t agentic-ai-workbench .
docker run -p 8501:8501 agentic-ai-workbench
```

Then open the app at:

http://localhost:8501

## Dependencies

The repository uses:

- `streamlit`
- `langgraph`
- `langchain-core`
- `langchain-openai`
- `langchain-community`
- `langchain-text-splitters`
- `langchain-tavily`
- `tavily-python`
- `langchain-huggingface`
- `sentence-transformers`
- `faiss-cpu`
- `pypdf`
- `pydantic`
- `python-dotenv`
- `requests`

## Notes

This project is a demonstration of an end-to-end agentic AI assistant pattern. It uses mock purchase handling and external APIs where keys are configured, so it is suitable for local development, learning, and experimentation with agentic workflows, RAG, and HITL decision points.
