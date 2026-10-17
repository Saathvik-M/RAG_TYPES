"""Session state and helper utilities."""
import streamlit as st


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None
    if "last_response" not in st.session_state:
        st.session_state.last_response = None


def add_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.last_response = None