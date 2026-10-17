"""FastAPI / backend client calls."""
import requests
import streamlit as st

API_URL = "http://localhost:8000"


def process_pdf(uploaded_file) -> bool:
    """Send PDF to backend for parsing/chunking/embedding."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_URL}/upload", files=files, timeout=120)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Make sure the FastAPI server is running on port 8000.")
        return False
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        return False


def _call_chat_api(query: str, rag_type: str) -> dict:
    """Base helper to call the /chat endpoint and return the full JSON response."""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"question": query, "rag_type": rag_type},
            timeout=60,
        )
        if response.status_code == 200:
            try:
                data = response.json()
            except Exception:
                return {"answer": "Backend returned invalid JSON.", "chunks": [], "model": "N/A", "time_taken": 0.0, "rag": rag_type}
            return {
                "answer": data.get("answer", "No answer returned."),
                "chunks": data.get("chunks", []),
                "model": data.get("model", "N/A"),
                "time_taken": data.get("time_taken", 0.0),
                "rag": data.get("rag", rag_type),
            }
        elif response.status_code == 400:
            try:
                detail = response.json().get('detail', 'Bad request')
            except Exception:
                detail = response.text or 'Bad request'
            return {"answer": f"Error: {detail}", "chunks": [], "model": "N/A", "time_taken": 0.0, "rag": rag_type}
        else:
            try:
                detail = response.json().get('detail', 'Unknown error')
            except Exception:
                detail = response.text or 'Unknown error'
            return {"answer": f"Backend error (status {response.status_code}): {detail}", "chunks": [], "model": "N/A", "time_taken": 0.0, "rag": rag_type}
    except requests.exceptions.ConnectionError:
        return {"answer": "Cannot connect to backend. Make sure the FastAPI server is running on port 8000.", "chunks": [], "model": "N/A", "time_taken": 0.0, "rag": rag_type}
    except Exception as e:
        return {"answer": f"Request failed: {str(e)}", "chunks": [], "model": "N/A", "time_taken": 0.0, "rag": rag_type}


def run_naive_rag(query: str, history: list) -> dict:
    """Single-pass retrieve + generate."""
    return _call_chat_api(query, "naive")


def run_agentic_rag(query: str, history: list) -> dict:
    """Multi-step agent: plan, tool calls, synthesize."""
    return _call_chat_api(query, "agentic")


def run_self_correcting_rag(query: str, history: list) -> dict:
    """Generate, critique against context, revise."""
    return _call_chat_api(query, "self")


def run_advanced_rag(query: str, history: list) -> dict:
    """Advanced RAG with re-ranking."""
    return _call_chat_api(query, "advanced")


def run_hybrid_rag(query: str, history: list) -> dict:
    """Hybrid RAG combining keyword and semantic search."""
    return _call_chat_api(query, "hybrid")


def run_graph_rag(query: str, history: list) -> dict:
    """Graph-based RAG."""
    return _call_chat_api(query, "graph")


STRATEGY_FUNCS = {
    "Naive RAG": run_naive_rag,
    "Advanced RAG": run_advanced_rag,
    "Hybrid RAG": run_hybrid_rag,
    "Graph RAG": run_graph_rag,
    "Agentic RAG": run_agentic_rag,
    "Self-Correcting RAG": run_self_correcting_rag,
}


def route_query(strategy: str, query: str, history: list) -> dict:
    """Dispatch query to the selected RAG strategy handler."""
    handler = STRATEGY_FUNCS.get(strategy)
    if handler is None:
        return {"answer": f"Unknown strategy: {strategy}", "chunks": [], "model": "N/A", "time_taken": 0.0, "rag": strategy}
    return handler(query, history)