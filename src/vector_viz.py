from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import pinecone_store

LIST_PAGE_SIZE = 100
FETCH_BATCH = 100


@dataclass
class VectorSpaceResult:
    x: list[float]
    y: list[float]
    cluster: list[int]
    text: list[str]
    source: list[str]
    page: list[int | None]
    n_vectors: int


def fetch_all_vectors() -> tuple[np.ndarray, list[dict]]:
    """Pull every vector + metadata out of the configured Pinecone index."""
    index = pinecone_store._index()

    ids: list[str] = []
    for page in index.list(limit=LIST_PAGE_SIZE):
        ids.extend(item.id for item in page.vectors)

    vectors: list[list[float]] = []
    metas: list[dict] = []
    for i in range(0, len(ids), FETCH_BATCH):
        batch_ids = ids[i:i + FETCH_BATCH]
        result = index.fetch(ids=batch_ids)
        for vec_id in batch_ids:
            rec = result.vectors.get(vec_id)
            if rec is None:
                continue
            vectors.append(rec.values)
            metas.append(dict(rec.metadata or {}))

    return np.array(vectors, dtype=np.float32), metas


def _reduce_and_cluster(vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE

    coords = TSNE(
        n_components=2,
        init="pca",
        perplexity=min(30, max(5, len(vectors) // 100)),
        random_state=42,
    ).fit_transform(vectors)

    clusters = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(vectors)
    return coords, clusters


def build_vector_space(k: int) -> VectorSpaceResult:
    """Fetch the whole index, reduce to 2D (t-SNE), and cluster (k-means)."""
    vectors, metas = fetch_all_vectors()
    if len(vectors) == 0:
        raise RuntimeError("Pinecone index is empty — nothing to visualize.")

    coords, clusters = _reduce_and_cluster(vectors, k)

    texts, sources, pages = [], [], []
    for m in metas:
        texts.append((m.get("text") or "")[:160])
        sources.append(m.get("source", "unknown"))
        pages.append(m.get("page"))

    return VectorSpaceResult(
        x=coords[:, 0].tolist(),
        y=coords[:, 1].tolist(),
        cluster=[int(c) for c in clusters],
        text=texts,
        source=sources,
        page=pages,
        n_vectors=len(vectors),
    )
