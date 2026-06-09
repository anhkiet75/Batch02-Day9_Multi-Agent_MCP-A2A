"""LangGraph Supervisor-Workers orchestration for the Day 8 RAG pipeline."""

import operator
import logging
from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from src.task9_retrieval_pipeline import retrieve
Route = Literal["legal", "news", "mixed"]
LOGGER = logging.getLogger(__name__)
LEGAL_KEYWORDS = {
    "bộ luật", "điều luật", "hình phạt", "luật",
    "nghị định", "pháp luật", "quy định", "tội",
}
NEWS_KEYWORDS = {"bài báo", "bị bắt", "nghệ sĩ", "sự kiện", "tin tức"}

class RagAgentState(TypedDict, total=False):
    question: str
    top_k: int
    route: Route
    legal_results: Annotated[list[dict], operator.add]
    news_results: Annotated[list[dict], operator.add]
    worker_errors: Annotated[list[str], operator.add]
    evidence: list[dict]
    has_sufficient_evidence: bool
    answer: str

def classify_query(question: str) -> Route:
    """Classify a question for deterministic worker routing."""
    normalized = question.strip().lower()
    if not normalized:
        raise ValueError("Câu hỏi không được để trống")

    has_legal = any(keyword in normalized for keyword in LEGAL_KEYWORDS)
    has_news = any(keyword in normalized for keyword in NEWS_KEYWORDS)
    if has_legal and not has_news:
        return "legal"
    if has_news and not has_legal:
        return "news"
    return "mixed"


def supervisor(state: RagAgentState) -> dict:
    """Classify the request before dispatching specialist workers."""
    return {"route": classify_query(state["question"])}


def route_workers(state: RagAgentState) -> list[Send]:
    """Dispatch the workers selected by the Supervisor."""
    worker_state = {"question": state["question"], "top_k": state["top_k"]}
    if state["route"] == "legal":
        return [Send("legal_worker", worker_state)]
    if state["route"] == "news":
        return [Send("news_worker", worker_state)]
    return [
        Send("legal_worker", worker_state),
        Send("news_worker", worker_state),
    ]


def _infer_document_type(metadata: dict) -> str:
    doc_type = str(metadata.get("type", "")).lower()
    if doc_type in {"legal", "news"}:
        return doc_type

    location = " ".join(
        str(metadata.get(key, "")).lower()
        for key in ("source", "filename", "path")
    )
    if any(marker in location for marker in ("article", "/news/", "news_")):
        return "news"
    if any(
        marker in location
        for marker in ("bo_luat", "luat_", "nghi_dinh", "/legal/", "legal_")
    ):
        return "legal"
    return ""


def _retrieve_domain(state: dict, domain: str) -> dict:
    result_key = f"{domain}_results"
    try:
        top_k = max(int(state.get("top_k", 5)), 1)
        candidates = retrieve(state["question"], top_k=top_k * 3)
        results = []
        for item in candidates:
            metadata = item.get("metadata", {})
            doc_type = _infer_document_type(metadata)
            if doc_type != domain:
                continue
            metadata = {**metadata, "type": doc_type}
            if not metadata.get("source") and metadata.get("filename"):
                metadata["source"] = metadata["filename"]
            item = {**item, "metadata": metadata}
            results.append(item)
            if len(results) == top_k:
                break
        return {result_key: results, "worker_errors": []}
    except Exception as exc:
        LOGGER.exception("%s retrieval worker failed", domain, exc_info=exc)
        return {result_key: [], "worker_errors": [f"{domain}: retrieval_failed"]}


def legal_worker(state: dict) -> dict:
    """Retrieve evidence from legal documents."""
    return _retrieve_domain(state, "legal")


def news_worker(state: dict) -> dict:
    """Retrieve evidence from news documents."""
    return _retrieve_domain(state, "news")


def _evidence_key(item: dict) -> tuple[str, str]:
    metadata = item.get("metadata", {})
    source = metadata.get("source") or metadata.get("filename") or ""
    return str(source), item.get("content", "").strip()


def evidence_worker(state: dict) -> dict:
    """Merge worker evidence, remove duplicates, and prepare LLM ordering."""
    best_by_key: dict[tuple[str, str], dict] = {}
    candidates = state.get("legal_results", []) + state.get("news_results", [])
    for item in candidates:
        key = _evidence_key(item)
        current = best_by_key.get(key)
        if current is None or float(item.get("score", 0.0)) > float(
            current.get("score", 0.0)
        ):
            best_by_key[key] = item

    score = lambda item: float(item.get("score", 0.0))
    evidence = sorted(best_by_key.values(), key=score, reverse=True)
    top_k = max(int(state.get("top_k", len(evidence) or 1)), 1)
    evidence = evidence[:top_k]
    if evidence:
        from src.task10_generation import reorder_for_llm

        evidence = reorder_for_llm(evidence)
    return {"evidence": evidence, "has_sufficient_evidence": bool(evidence)}


def aggregator(state: RagAgentState) -> dict:
    """Generate the final answer from evidence validated by the third worker."""
    if not state.get("has_sufficient_evidence"):
        return {"answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có."}

    from src.task10_generation import generate_answer_from_chunks

    answer = generate_answer_from_chunks(state["question"], state["evidence"])
    return {"answer": answer}


@lru_cache(maxsize=1)
def build_supervisor_graph():
    """Build and compile the Stage 4 in-process graph."""
    graph = StateGraph(RagAgentState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("legal_worker", legal_worker)
    graph.add_node("news_worker", news_worker)
    graph.add_node("evidence_worker", evidence_worker)
    graph.add_node("aggregator", aggregator)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", route_workers, ["legal_worker", "news_worker"]
    )
    graph.add_edge("legal_worker", "evidence_worker")
    graph.add_edge("news_worker", "evidence_worker")
    graph.add_edge("evidence_worker", "aggregator")
    graph.add_edge("aggregator", END)
    return graph.compile()


def run_supervisor_workers(question: str, top_k: int = 5) -> dict:
    """Run the Supervisor and three workers, returning the public RAG result."""
    result = build_supervisor_graph().invoke(
        {
            "question": question,
            "top_k": max(int(top_k), 1),
            "legal_results": [],
            "news_results": [],
            "worker_errors": [],
            "evidence": [],
            "has_sufficient_evidence": False,
            "answer": "",
        }
    )
    sources = result.get("evidence", [])
    retrieval_source = sources[0].get("source", "none") if sources else "none"
    return {
        "answer": result["answer"],
        "sources": sources,
        "retrieval_source": retrieval_source,
        "route": result["route"],
        "worker_errors": result.get("worker_errors", []),
    }
