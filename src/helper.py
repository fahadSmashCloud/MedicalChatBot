from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

AVAILABLE_MODELS = {
    "Llama 3.3 70B (versatile)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (instant)": "llama-3.1-8b-instant",
    "Gemma 2 9B": "gemma2-9b-it",
    "Llama 3 70B (legacy)": "llama3-70b-8192",
}


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_vectorstore() -> FAISS:
    return FAISS.load_local(
        DB_FAISS_PATH,
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )


def save_vectorstore(db: FAISS) -> None:
    db.save_local(DB_FAISS_PATH)


def chunk_documents(docs: Iterable[Document], chunk_size: int = 500, overlap: int = 50) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(list(docs))


def load_pdfs_from_dir(data_dir: str) -> list[Document]:
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


class PDFIngestError(RuntimeError):
    """Raised when an uploaded PDF cannot be ingested (no extractable text, encrypted, etc.)."""


def ingest_uploaded_pdfs(uploaded_files) -> int:
    """Append uploaded PDFs to the existing FAISS index. Returns number of new chunks added.

    Raises PDFIngestError if no PDF produced extractable text.
    """
    if not uploaded_files:
        return 0

    new_docs: list[Document] = []
    failed: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for f in uploaded_files:
            target = tmp_path / f.name
            target.write_bytes(f.getvalue())
            try:
                loaded = PyPDFLoader(str(target)).load()
            except Exception as e:
                failed.append(f"{f.name} ({e})")
                continue
            non_empty = [d for d in loaded if (d.page_content or "").strip()]
            if not non_empty:
                failed.append(f"{f.name} (no extractable text — likely scanned or image-only)")
            else:
                new_docs.extend(non_empty)

    if not new_docs:
        raise PDFIngestError(
            "No text could be extracted from the uploaded PDF(s): "
            + "; ".join(failed)
        )

    chunks = chunk_documents(new_docs)
    if not chunks:
        raise PDFIngestError("PDFs loaded but text-splitting produced no chunks.")

    embeddings = get_embedding_model()

    if Path(DB_FAISS_PATH).exists():
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        new_db = FAISS.from_documents(chunks, embeddings)
        db.merge_from(new_db)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    save_vectorstore(db)
    return len(chunks)


def build_llm(model_id: str, temperature: float, api_key: str, streaming: bool = True) -> ChatGroq:
    return ChatGroq(
        model=model_id,
        temperature=temperature,
        api_key=api_key,
        streaming=streaming,
    )


def format_docs_with_sources(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        page = d.metadata.get("page")
        src = d.metadata.get("source", "unknown")
        header = f"[Source {i} — {Path(src).name}" + (f", p.{page + 1}" if page is not None else "") + "]"
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n".join(parts)


def build_chain(
    system_prompt: str,
    llm: ChatGroq,
    retriever,
):
    """Returns a runnable that takes {'question': str, 'chat_history': [(human, ai), ...]} and streams a string."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    def _retrieve(inputs: dict) -> dict:
        docs = retriever.invoke(inputs["question"])
        return {
            "context": format_docs_with_sources(docs),
            "question": inputs["question"],
            "chat_history": inputs.get("chat_history", []),
            "_docs": docs,
        }

    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: x["context"])
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableLambda(_retrieve), answer_chain


def history_as_messages(messages: list[dict], max_turns: int = 4):
    """Convert Streamlit-style message dicts into LangChain (role, content) tuples, keeping the last N turns."""
    from langchain_core.messages import AIMessage, HumanMessage
    recent = messages[-(max_turns * 2):]
    out = []
    for m in recent:
        if m["role"] == "user":
            out.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            out.append(AIMessage(content=m["content"]))
    return out
