"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import json
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.task4_chunking_indexing import (
    ARTIFACT_DIR,
    EMBEDDING_DIM,
    VECTOR_STORE,
    WEAVIATE_COLLECTION,
    normalize_weaviate_url,
    chunk_documents,
    embed_chunks,
    load_documents,
)

import atexit
import threading

INDEX_PATH = ARTIFACT_DIR / "task4_chunks_index.json"
_CHUNK_CACHE: list[dict] | None = None
_weaviate_client = None
_weaviate_lock = threading.Lock()


def _fallback_embed_text(text: str) -> list[float]:
    text = text[:EMBEDDING_DIM]
    vector = [float((ord(char) % 256) / 255.0) for char in text]
    if len(vector) < EMBEDDING_DIM:
        vector.extend([0.0] * (EMBEDDING_DIM - len(vector)))
    return vector[:EMBEDDING_DIM]


def _embed_query(query: str) -> list[float]:
    return _fallback_embed_text(query)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _load_index() -> list[dict]:
    global _CHUNK_CACHE

    if _CHUNK_CACHE is not None:
        return _CHUNK_CACHE

    if INDEX_PATH.exists():
        _CHUNK_CACHE = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        return _CHUNK_CACHE

    documents = load_documents()
    chunks = chunk_documents(documents)
    _CHUNK_CACHE = embed_chunks(chunks)
    return _CHUNK_CACHE


def _get_weaviate_client():
    """Return a shared Weaviate client, creating one if needed. Reused across queries to avoid per-query TCP overhead."""
    global _weaviate_client
    if _weaviate_client is not None:
        return _weaviate_client

    with _weaviate_lock:
        if _weaviate_client is not None:
            return _weaviate_client

        import weaviate
        from weaviate.auth import AuthApiKey

        weaviate_url = os.getenv("WEAVIATE_URL", "")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY", "")
        if not weaviate_url or not weaviate_api_key:
            raise RuntimeError("Thiếu WEAVIATE_URL hoặc WEAVIATE_API_KEY để query Weaviate Cloud")

        _weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=normalize_weaviate_url(weaviate_url),
            auth_credentials=AuthApiKey(weaviate_api_key),
            skip_init_checks=True,
        )
        atexit.register(lambda: _weaviate_client.close() if _weaviate_client else None)
    return _weaviate_client


def _semantic_search_weaviate(query: str, top_k: int) -> list[dict]:
    from weaviate.classes.query import MetadataQuery

    query_embedding = _embed_query(query)
    client = _get_weaviate_client()
    collection = client.collections.get(WEAVIATE_COLLECTION)
    results = collection.query.near_vector(
        near_vector=query_embedding,
        limit=top_k,
        return_metadata=MetadataQuery(distance=True),
    )
    return [
        {
            "content": obj.properties.get("content", ""),
            "score": float(1.0 - getattr(obj.metadata, "distance", 1.0)),
            "metadata": {
                "source": obj.properties.get("source"),
                "path": obj.properties.get("path"),
                "type": obj.properties.get("doc_type"),
                "chunk_index": obj.properties.get("chunk_index"),
            },
        }
        for obj in results.objects
    ]


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if top_k <= 0:
        return []
    if VECTOR_STORE == "weaviate":
        try:
            return _semantic_search_weaviate(query, top_k)
        except Exception as exc:
            print(f"⚠ Weaviate query failed, fallback to local index: {exc}")

    chunks = _load_index()
    if not chunks:
        return []

    query_embedding = _embed_query(query)
    scored = []
    for chunk in chunks:
        score = _cosine_similarity(query_embedding, chunk["embedding"])
        scored.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk["metadata"],
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
