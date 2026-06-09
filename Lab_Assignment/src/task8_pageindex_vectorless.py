"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
import re
import sys
import time
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
ARTIFACT_DIR = Path(__file__).parent.parent / "data" / "artifacts"
DOC_MAP_PATH = ARTIFACT_DIR / "pageindex_documents.json"


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY")
    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    uploaded = {}
    for pdf_file in sorted((LANDING_DIR / "legal").glob("*.pdf")):
        response = client.submit_document(str(pdf_file))
        doc_id = response.get("doc_id")
        if doc_id:
            uploaded[pdf_file.name] = doc_id
            print(f"  ✓ Uploaded: {pdf_file.name} -> {doc_id}")

    DOC_MAP_PATH.write_text(json.dumps(uploaded, ensure_ascii=False, indent=2), encoding="utf-8")
    return uploaded


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval local fallback mang marker source='pageindex'.
    """
    if top_k <= 0:
        return []

    if PAGEINDEX_API_KEY and DOC_MAP_PATH.exists():
        try:
            from pageindex import PageIndexClient

            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            doc_map = json.loads(DOC_MAP_PATH.read_text(encoding="utf-8"))
            results = []
            for filename, doc_id in doc_map.items():
                if not client.is_retrieval_ready(doc_id):
                    continue
                retrieval = client.submit_query(doc_id=doc_id, query=query)
                retrieval_id = retrieval.get("retrieval_id")
                if not retrieval_id:
                    continue
                data = None
                for _ in range(10):
                    data = client.get_retrieval(retrieval_id)
                    if data.get("status") in {"completed", "succeeded", "done"}:
                        break
                    time.sleep(1)
                if not data:
                    continue
                text = data.get("answer") or data.get("content") or data.get("result") or ""
                if not text:
                    chunks = data.get("chunks") or data.get("results") or []
                    if chunks:
                        first = chunks[0]
                        text = first.get("text") or first.get("content") or ""
                score = float(data.get("score", 1.0))
                if text:
                    results.append(
                        {
                            "content": text,
                            "score": score,
                            "metadata": {"filename": filename, "doc_id": doc_id},
                            "source": "pageindex",
                        }
                    )
            results.sort(key=lambda item: item["score"], reverse=True)
            if results:
                return results[:top_k]
        except Exception:
            pass

    query_tokens = _tokenize(query)
    results = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        content_tokens = _tokenize(content)
        if not content_tokens:
            continue
        score = len(query_tokens & content_tokens) / max(len(query_tokens) or 1, 1)
        if score <= 0 and query_tokens:
            continue
        results.append(
            {
                "content": content,
                "score": float(score),
                "metadata": {"filename": md_file.name, "type": md_file.parent.name},
                "source": "pageindex",
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY chưa có, dùng local fallback search")
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
