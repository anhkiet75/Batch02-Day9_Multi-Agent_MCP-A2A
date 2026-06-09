"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import math
import re
from typing import Optional


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _score_overlap(query: str, content: str) -> float:
    query_tokens = _tokenize(query)
    content_tokens = _tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0
    return len(query_tokens & content_tokens) / len(query_tokens)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng heuristic overlap local-safe.
    """
    rescored = []
    for candidate in candidates:
        overlap_score = _score_overlap(query, candidate["content"])
        final_score = 0.7 * overlap_score + 0.3 * float(candidate.get("score", 0.0))
        rescored.append({**candidate, "score": float(final_score)})
    rescored.sort(key=lambda item: item["score"], reverse=True)
    return rescored[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.
    """
    if not candidates or top_k <= 0:
        return []

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")
        for idx in remaining:
            candidate_embedding = candidates[idx].get("embedding", [])
            relevance = _cosine_similarity(query_embedding, candidate_embedding) if candidate_embedding else float(candidates[idx].get("score", 0.0))
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                selected_embedding = candidates[sel_idx].get("embedding", [])
                if candidate_embedding and selected_embedding:
                    sim = _cosine_similarity(candidate_embedding, selected_embedding)
                    max_sim_to_selected = max(max_sim_to_selected, sim)
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        if best_idx is None:
            break
        item = {**candidates[best_idx], "score": float(best_score)}
        selected.append(best_idx)
        remaining.remove(best_idx)
        candidates[best_idx] = item

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.
    """
    rrf_scores = {}
    item_map = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            item_map[key] = item

    merged = []
    for content, score in sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]:
        merged.append({**item_map[content], "score": float(score)})
    return merged


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        query_embedding = candidates[0].get("embedding", []) if candidates else []
        return rerank_mmr(query_embedding, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
