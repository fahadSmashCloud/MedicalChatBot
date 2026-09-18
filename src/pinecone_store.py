from __future__ import annotations

import os
import uuid

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

UPSERT_BATCH = 100


def is_configured() -> bool:
    return bool(os.environ.get("PINECONE_API_KEY"))


def index_name() -> str:
    return os.environ.get("PINECONE_INDEX_NAME", "medicalbot")


def _index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    host = os.environ.get("PINECONE_INDEX_HOST")
    return pc.Index(host=host) if host else pc.Index(index_name())


class PineconeRetriever:
    """Runnable-shaped retriever: .invoke(query) -> list[Document], matching FAISS's as_retriever() output."""

    def __init__(self, index, embeddings: HuggingFaceEmbeddings, k: int):
        self._index = index
        self._embeddings = embeddings
        self._k = k

    def invoke(self, query: str) -> list[Document]:
        vector = self._embeddings.embed_query(query)
        result = self._index.query(vector=vector, top_k=self._k, include_metadata=True)
        docs = []
        for match in result.matches:
            meta = dict(match.metadata or {})
            text = meta.pop("text", "")
            docs.append(Document(page_content=text, metadata=meta))
        return docs


class PineconeVectorStore:
    """FAISS-shaped wrapper (as_retriever / add_documents) so medibot.py's call sites don't branch on backend."""

    def __init__(self, embeddings: HuggingFaceEmbeddings):
        self._embeddings = embeddings
        self._index = _index()

    def as_retriever(self, search_kwargs: dict | None = None) -> PineconeRetriever:
        k = (search_kwargs or {}).get("k", 4)
        return PineconeRetriever(self._index, self._embeddings, k)

    def chunk_count(self) -> int:
        stats = self._index.describe_index_stats()
        return int(stats.total_vector_count)

    def add_documents(self, docs: list[Document]) -> int:
        texts = [d.page_content for d in docs]
        vectors_out = self._embeddings.embed_documents(texts)
        records = []
        for doc, vector in zip(docs, vectors_out):
            meta = {k: v for k, v in doc.metadata.items() if v is not None}
            meta["text"] = doc.page_content
            records.append({"id": str(uuid.uuid4()), "values": vector, "metadata": meta})

        for i in range(0, len(records), UPSERT_BATCH):
            self._index.upsert(vectors=records[i:i + UPSERT_BATCH])
        return len(records)
