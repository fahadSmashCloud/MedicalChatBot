from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import find_dotenv, load_dotenv

from src.helper import (
    AVAILABLE_MODELS,
    PDFIngestError,
    build_chain,
    build_llm,
    history_as_messages,
    ingest_uploaded_pdfs,
    load_vectorstore,
)
from src.prompt import ASSISTED_SYSTEM_PROMPT, STRICT_SYSTEM_PROMPT, SUGGESTED_QUESTIONS

load_dotenv(find_dotenv())

st.set_page_config(
    page_title="MediBot — Medical Reference Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- styling ----------
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .subtle { color: #888; font-size: 0.9rem; }
    .disclaimer {
        background: #fff4e5; border-left: 4px solid #ff9800;
        padding: 0.6rem 0.9rem; border-radius: 4px; font-size: 0.85rem; color: #5d3a00;
        margin-bottom: 1rem;
    }
    .chip {
        display: inline-block; padding: 4px 10px; margin: 3px;
        background: #eef3ff; border-radius: 999px; font-size: 0.85rem;
        border: 1px solid #d6e0ff;
    }
    .source-box {
        background: #f7f7f9; border-radius: 6px; padding: 8px 12px;
        font-size: 0.85rem; margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- session state ----------
def init_state():
    defaults = {
        "messages": [],
        "model_label": next(iter(AVAILABLE_MODELS)),
        "temperature": 0.4,
        "top_k": 3,
        "strict_mode": True,
        "memory_turns": 3,
        "last_sources": [],
        "last_usage": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


@st.cache_resource(show_spinner=False)
def cached_vectorstore():
    return load_vectorstore()


# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")

    st.session_state.model_label = st.selectbox(
        "Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.model_label),
    )
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.1)
    st.session_state.top_k = st.slider("Retrieved chunks (top-k)", 1, 8, st.session_state.top_k)
    st.session_state.memory_turns = st.slider("Conversation memory (turns)", 0, 8, st.session_state.memory_turns)
    st.session_state.strict_mode = st.toggle(
        "Strict context-only mode",
        value=st.session_state.strict_mode,
        help="When on, MediBot only answers from your uploaded reference material. When off, it may supplement with general medical knowledge.",
    )

    st.divider()
    st.subheader("📄 Add documents")
    uploads = st.file_uploader(
        "Upload medical PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Files are chunked and merged into the FAISS index.",
    )
    if uploads and st.button("Ingest uploaded PDFs", use_container_width=True):
        try:
            with st.spinner(f"Embedding {len(uploads)} file(s)…"):
                added = ingest_uploaded_pdfs(uploads)
                cached_vectorstore.clear()
            st.success(f"Added {added} chunks to the index.")
        except PDFIngestError as e:
            st.error(f"Ingest failed: {e}")
        except Exception as e:
            st.error(f"Unexpected error during ingest: {e}")

    st.divider()
    st.subheader("💬 Chat")
    c1, c2 = st.columns(2)
    if c1.button("🧹 Clear", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_usage = None
        st.rerun()
    if st.session_state.messages:
        transcript = "\n\n".join(
            f"**{m['role'].title()}**: {m['content']}" for m in st.session_state.messages
        )
        c2.download_button(
            "⬇️ Export",
            data=f"# MediBot chat — {datetime.now():%Y-%m-%d %H:%M}\n\n{transcript}",
            file_name=f"medibot-chat-{datetime.now():%Y%m%d-%H%M%S}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()
    try:
        db = cached_vectorstore()
        n_docs = db.index.ntotal
        st.caption(f"📚 {n_docs:,} indexed chunks")
    except Exception:
        st.caption("📚 Index not loaded")
    st.caption(f"🤖 Model: `{AVAILABLE_MODELS[st.session_state.model_label]}`")


# ---------- header ----------
st.markdown('<div class="main-header">🩺 MediBot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Retrieval-augmented medical reference chatbot — powered by Groq + FAISS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="disclaimer">⚠️ <strong>Not medical advice.</strong> '
    'MediBot is for educational reference only. Always consult a qualified healthcare professional '
    'for diagnosis, treatment, or medication decisions.</div>',
    unsafe_allow_html=True,
)


# ---------- guardrails ----------
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("**GROQ_API_KEY is not set.** Get a free key at https://console.groq.com/keys and add it to your `.env` file.")
    st.stop()

try:
    vectorstore = cached_vectorstore()
except Exception as e:
    st.error(f"Could not load FAISS index at `vectorstore/db_faiss`. Run `python app.py` first.\n\n{e}")
    st.stop()


# ---------- welcome / chat replay ----------
if not st.session_state.messages:
    st.subheader("Try a starter question")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 2].button(q, key=f"starter-{i}", use_container_width=True):
            st.session_state._queued_prompt = q
            st.rerun()
else:
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🩺"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])


# ---------- run a query ----------
queued = st.session_state.pop("_queued_prompt", None)
user_input = st.chat_input("Ask a medical reference question…") or queued

if user_input:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    system_prompt = STRICT_SYSTEM_PROMPT if st.session_state.strict_mode else ASSISTED_SYSTEM_PROMPT
    model_id = AVAILABLE_MODELS[st.session_state.model_label]
    llm = build_llm(model_id, st.session_state.temperature, groq_api_key, streaming=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": st.session_state.top_k})
    retrieve_step, answer_chain = build_chain(system_prompt, llm, retriever)

    chat_history = history_as_messages(
        st.session_state.messages[:-1], max_turns=st.session_state.memory_turns
    )

    with st.chat_message("assistant", avatar="🩺"):
        try:
            with st.spinner("Retrieving relevant context…"):
                retrieved = retrieve_step.invoke({
                    "question": user_input,
                    "chat_history": chat_history,
                })
            st.session_state.last_sources = retrieved["_docs"]

            stream = answer_chain.stream({
                "context": retrieved["context"],
                "question": retrieved["question"],
                "chat_history": retrieved["chat_history"],
            })
            full_response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.session_state.messages.pop()  # remove the user message so they can retry
            st.stop()

    # sources panel
    if st.session_state.last_sources:
        with st.expander(f"📖 Sources ({len(st.session_state.last_sources)} chunks)"):
            for i, doc in enumerate(st.session_state.last_sources, start=1):
                src = Path(doc.metadata.get("source", "unknown")).name
                page = doc.metadata.get("page")
                page_str = f", page {page + 1}" if page is not None else ""
                st.markdown(f"**Source {i}** — `{src}`{page_str}")
                st.markdown(f'<div class="source-box">{doc.page_content[:600]}{"…" if len(doc.page_content) > 600 else ""}</div>', unsafe_allow_html=True)
