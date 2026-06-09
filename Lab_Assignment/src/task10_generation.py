"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source.

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.
"""


def _extract_year(chunk: dict) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source", "")
    for token in source.replace(".md", "").split("_"):
        if token.isdigit() and len(token) == 4:
            return token
    content = chunk.get("content", "")
    for part in content.split():
        digits = "".join(ch for ch in part if ch.isdigit())
        if len(digits) == 4 and digits.startswith(("19", "20")):
            return digits
    return "n.d."


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh lost in the middle.
    """
    if len(chunks) <= 2:
        return chunks
    front = chunks[::2]
    back = chunks[1::2][::-1]
    return front + back


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        year = _extract_year(chunk)
        context_parts.append(
            f"[Document {i} | Source: {source} | Year: {year} | Type: {doc_type}]\n{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


def _fallback_generate_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I cannot verify this information."
    lines = []
    for chunk in chunks[: min(3, len(chunks))]:
        source = chunk.get("metadata", {}).get("source", "Nguồn không rõ")
        year = _extract_year(chunk)
        snippet = chunk["content"].strip().replace("\n", " ")
        lines.append(f"- {snippet[:220]} [{source}, {year}]")
    return (
        f"Dưới đây là các đoạn liên quan nhất cho câu hỏi: {query}\n"
        + "\n".join(lines)
    )


def generate_answer_from_chunks(query: str, chunks: list[dict]) -> str:
    """Generate a cited answer from evidence selected by the agent graph."""
    context = format_context(chunks)
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if openai_api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_api_key)
            user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content or ""
        except Exception:
            answer = _fallback_generate_answer(query, chunks)
    else:
        answer = _fallback_generate_answer(query, chunks)
    return answer


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Run the Supervisor-Workers RAG pipeline with citation generation."""
    from src.supervisor_workers import run_supervisor_workers

    return run_supervisor_workers(query, top_k=top_k)


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
