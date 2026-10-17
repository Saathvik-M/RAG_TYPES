"""Reusable Streamlit UI components."""
import streamlit as st
from api import process_pdf


# ---------------------------------------------------------------------------
# RAG Strategy descriptions with code-level examples from the backend
# ---------------------------------------------------------------------------
RAG_STRATEGY_INFO = {
    "Naive RAG": {
        "description": "Single-pass retrieve-and-generate. The question is used directly to retrieve relevant chunks, which are then passed to the LLM as context.",
        "code_example": """# backend/rag/naive.py
docs = self.retriever.retrieve(question)
context = "\\n\\n".join(doc.page_content for doc in docs)
prompt = PromptBuilder.build(context=context, question=question)
answer = self.llm.generate(prompt)""",
    },
    "Advanced RAG": {
        "description": "Enhanced retrieval (e.g., MMR for diversity) followed by the same generate step. Designed to improve chunk quality and reduce redundancy.",
        "code_example": """# backend/rag/advanced.py
docs = self.retriever.retrieve(question)
context = "\\n\\n".join(doc.page_content for doc in docs)
prompt = PromptBuilder.build(context=context, question=question)
answer = self.llm.generate(prompt)""",
    },
    "Hybrid RAG": {
        "description": "Combines keyword-based and semantic (vector) search to retrieve chunks, balancing exact-match and similarity-based results.",
        "code_example": """# backend/rag/hybrid.py
docs = self.retriever.retrieve(question)  # keyword-heavy approach
context = "\\n\\n".join(doc.page_content for doc in docs)
prompt = PromptBuilder.build(context=context, question=question)
answer = self.llm.generate(prompt)""",
    },
    "Graph RAG": {
        "description": "Uses relationship-aware retrieval. The LLM is prompted to analyze connections between pieces of information in the context.",
        "code_example": (
            "# backend/rag/graph.py\n"
            "docs = self.retriever.retrieve(question)\n"
            'context = "\\n\\n".join(doc.page_content for doc in docs)\n'
            "prompt = f'''\n"
            "You are an AI assistant with graph-based reasoning capabilities.\n"
            "Analyze the following context and answer the question. If relationships\n"
            "exist between different pieces of information, explain them clearly.\n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Answer:\n"
            "'''\n"
            "answer = self.llm.generate(prompt)"
        ),
    },
    "Agentic RAG": {
        "description": "Multi-step reasoning. The question is decomposed into sub-queries, each is retrieved separately, results are deduplicated, then a synthesis step produces the final answer.",
        "code_example": """# backend/rag/agentic.py
# Step 1: Decompose question into sub-queries
search_queries = self.llm.generate(planning_prompt)
# Step 2: Retrieve for each sub-query (deduplicated)
for sq in search_queries[:3]:
    docs = self.retriever.retrieve(sq)
# Step 3: Synthesize final answer
answer = self.llm.generate(synthesis_prompt)""",
    },
    "Self-Correcting RAG": {
        "description": "Generate an initial answer, then have the LLM critique it against the context. If unsupported claims are found, the answer is revised to be fully supported.",
        "code_example": """# backend/rag/self_rag.py
# Step 1: Generate initial answer
initial_answer = self.llm.generate(initial_prompt)
# Step 2: Self-critique
critique = self.llm.generate(critique_prompt)
# Step 3: Revise if unsupported claims found
if "fully supported" not in critique.lower():
    final_answer = self.llm.generate(revision_prompt)""",
    },
}


def render_sidebar() -> str:
    with st.sidebar:
        st.header("⚙️ Settings")

        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file is not None:
            if (st.session_state.pdf_name != uploaded_file.name or not st.session_state.pdf_processed):
                with st.spinner("Processing PDF..."):
                    success = process_pdf(uploaded_file)
                if success:
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.success(f"✅ {uploaded_file.name} processed!")
                else:
                    st.session_state.pdf_processed = False
                    st.session_state.pdf_name = None

        strategy = st.selectbox(
            "Choose RAG",
            ["Naive RAG", "Advanced RAG", "Hybrid RAG", "Graph RAG", "Agentic RAG", "Self-Correcting RAG"],
        )

        st.divider()

        # --- Model Information Section ---
        st.subheader("Model Information")
        if st.session_state.get("last_response"):
            last = st.session_state.last_response
            st.caption(f"**Model:** {last.get('model', 'N/A')}")
            st.caption(f"**RAG Type:** {last.get('rag', strategy)}")
        else:
            st.caption("No queries yet.")

        st.divider()

        # --- RAG Strategy Guide ---
        with st.expander("📖 RAG Strategy Guide", expanded=False):
            info = RAG_STRATEGY_INFO.get(strategy, {})
            st.markdown(f"**{strategy}**")
            st.markdown(info.get("description", ""))
            st.code(info.get("code_example", ""), language="python")

        st.divider()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_response = None
            st.rerun()

    return strategy


def render_pdf_status() -> None:
    if st.session_state.get("pdf_processed"):
        st.success(f"📄 Using document: {st.session_state.get('pdf_name', 'uploaded.pdf')}")
    else:
        st.info("📄 No document uploaded. Please upload a PDF to get started.")


def render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render_strategy_description(strategy: str) -> None:
    """Show a brief description of the selected RAG strategy above the question box."""
    info = RAG_STRATEGY_INFO.get(strategy, {})
    desc = info.get("description", "")
    if desc:
        st.info(f"ℹ️ **{strategy}:** {desc}")


def render_answer_and_details(response: dict) -> None:
    """Display assistant answer, retrieved chunks, and statistics."""
    answer = response.get("answer", "No answer returned.")
    chunks = response.get("chunks", [])
    model = response.get("model", "N/A")
    time_taken = response.get("time_taken", 0.0)
    rag = response.get("rag", "N/A")

    # --- Answer ---
    st.subheader("Answer")
    st.markdown(answer)

    # --- Retrieved Chunks ---
    st.divider()
    st.subheader("Retrieved Chunks")

    if chunks:
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "")
            page = chunk.get("page", "?")
            source = chunk.get("source", "unknown")
            with st.expander(f"▼ Chunk {i + 1} — Page {page} | {source}"):
                st.text(chunk_text)
    else:
        st.caption("No chunks retrieved.")

    # --- Statistics ---
    st.divider()
    st.subheader("Statistics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", model)
    with col2:
        st.metric("Time", f"{time_taken:.2f} seconds")
    with col3:
        st.metric("Pages Retrieved", len(chunks))