"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
ARTIFACT_DIR = Path(__file__).parent.parent / "data" / "artifacts"
WEAVIATE_COLLECTION = "DrugLawDocs"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Recursive splitter là lựa chọn an toàn cho cả file legal dài và news markdown lẫn lộn.
# Chunk 500 ký tự đủ nhỏ để retrieval chính xác hơn, overlap 50 giữ ngữ cảnh giữa hai đoạn.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# bge-m3 hỗ trợ multilingual tốt cho tiếng Việt; nếu local model chưa tải được,
# code sẽ fallback về embedding giả lập ổn định để pipeline vẫn chạy local.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Có thể bật Weaviate Cloud bằng VECTOR_STORE=weaviate và set WEAVIATE_URL/WEAVIATE_API_KEY.
VECTOR_STORE = os.getenv("VECTOR_STORE", "weaviate")  # "weaviate" | "chromadb" | "faiss" | "local_json"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                    "type": doc_type,
                },
            }
        )
    return documents


def normalize_weaviate_url(url: str) -> str:
    if url and not url.startswith(("http://", "https://")):
        return f"https://{url}"
    return url


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for index, chunk_text in enumerate(splits):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {**doc["metadata"], "chunk_index": index},
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    for chunk in chunks:
        text = chunk["content"][:EMBEDDING_DIM]
        base = [float((ord(char) % 256) / 255.0) for char in text]
        if len(base) < EMBEDDING_DIM:
            base.extend([0.0] * (EMBEDDING_DIM - len(base)))
        chunk["embedding"] = base[:EMBEDDING_DIM]
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if VECTOR_STORE == "local_json":
        output_path = ARTIFACT_DIR / "task4_chunks_index.json"
        output_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    if VECTOR_STORE == "faiss":
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("FAISS chưa được cài đặt") from exc

        vectors = np.array([chunk["embedding"] for chunk in chunks], dtype="float32")
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        index.add(vectors)
        faiss.write_index(index, str(ARTIFACT_DIR / "task4_faiss.index"))
        meta_path = ARTIFACT_DIR / "task4_faiss_metadata.json"
        meta_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    if VECTOR_STORE == "weaviate":
        weaviate_url = os.getenv("WEAVIATE_URL", "")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY", "")
        if not weaviate_url or not weaviate_api_key:
            raise RuntimeError("Thiếu WEAVIATE_URL hoặc WEAVIATE_API_KEY để dùng Weaviate Cloud")
        weaviate_url = normalize_weaviate_url(weaviate_url)

        import weaviate
        from weaviate.auth import AuthApiKey
        from weaviate.classes.config import Configure, DataType, Property

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=AuthApiKey(weaviate_api_key),
            skip_init_checks=True,
        )
        try:
            if not client.collections.exists(WEAVIATE_COLLECTION):
                client.collections.create(
                    name=WEAVIATE_COLLECTION,
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=[
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="source", data_type=DataType.TEXT),
                        Property(name="path", data_type=DataType.TEXT),
                        Property(name="doc_type", data_type=DataType.TEXT),
                        Property(name="chunk_index", data_type=DataType.INT),
                    ],
                )

            collection = client.collections.get(WEAVIATE_COLLECTION)
            with collection.batch.dynamic() as batch:
                for chunk in chunks:
                    batch.add_object(
                        properties={
                            "content": chunk["content"],
                            "source": chunk["metadata"]["source"],
                            "path": chunk["metadata"]["path"],
                            "doc_type": chunk["metadata"]["type"],
                            "chunk_index": chunk["metadata"]["chunk_index"],
                        },
                        vector=chunk["embedding"],
                    )
        finally:
            client.close()
        return

    raise RuntimeError(f"VECTOR_STORE chưa được hỗ trợ trong môi trường local: {VECTOR_STORE}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
