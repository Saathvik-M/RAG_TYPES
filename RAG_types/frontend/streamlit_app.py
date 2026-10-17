import streamlit as st
from components import (
    render_sidebar,
    render_chat_history,
    render_pdf_status,
    render_answer_and_details,
    render_strategy_description,
)
from utils import init_session_state, add_message
from api import route_query

st.set_page_config(
    page_title="RAG Playground",
    layout="wide"
)

st.title("📚 RAG Playground")

sidebar = st.sidebar
sidebar.header("Settings")

init_session_state()

strategy = render_sidebar()

st.caption(f"Strategy: **{strategy}**")

# Show a brief description of the selected RAG strategy
render_strategy_description(strategy)

render_pdf_status()
render_chat_history()

if prompt := st.chat_input("Ask a question about your document..."):
    add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = route_query(strategy, prompt, st.session_state.messages)
        render_answer_and_details(response)

    # Store the last response for the sidebar model info
    st.session_state.last_response = response

    # Also store the full answer in messages for chat history rendering
    add_message("assistant", response.get("answer", "No answer returned."))
