"""Tests for the in-process Supervisor-Workers RAG graph."""

import pytest

@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Hình phạt cho tội tàng trữ ma túy là gì?", "legal"),
        ("Nghệ sĩ nào bị bắt vì sử dụng ma túy?", "news"),
        ("Nghệ sĩ bị bắt sẽ chịu hình phạt nào?", "mixed"),
        ("Cho tôi thông tin về ma túy", "mixed"),
    ],
)
def test_classify_query_routes_supported_domains(question, expected):
    from src.supervisor_workers import classify_query
    assert classify_query(question) == expected

def test_classify_query_rejects_blank_question():
    from src.supervisor_workers import classify_query
    with pytest.raises(ValueError, match="không được để trống"):
        classify_query("   ")

def _chunk(content, score, doc_type, source="source.md"):
    return {
        "content": content,
        "score": score,
        "metadata": {"type": doc_type, "source": source},
        "source": "hybrid",
    }

def test_legal_worker_filters_non_legal_results(monkeypatch):
    import src.supervisor_workers as module
    monkeypatch.setattr(
        module,
        "retrieve",
        lambda query, top_k: [
            _chunk("Legal evidence", 0.9, "legal"),
            _chunk("News evidence", 0.8, "news"),
        ],
    )
    result = module.legal_worker({"question": "Luật ma túy", "top_k": 2})
    assert [item["content"] for item in result["legal_results"]] == [
        "Legal evidence"
    ]
    assert result["worker_errors"] == []

def test_news_worker_captures_retrieval_failure(monkeypatch):
    import src.supervisor_workers as module
    def fail_retrieval(query, top_k):
        raise RuntimeError("index unavailable")
    monkeypatch.setattr(module, "retrieve", fail_retrieval)
    result = module.news_worker({"question": "Tin nghệ sĩ", "top_k": 2})
    assert result["news_results"] == []
    assert result["worker_errors"] == ["news: retrieval_failed"]

def test_legal_worker_accepts_pageindex_filename_without_type(monkeypatch):
    import src.supervisor_workers as module
    monkeypatch.setattr(
        module,
        "retrieve",
        lambda query, top_k: [
            {
                "content": "Legal PageIndex evidence",
                "score": 0.9,
                "metadata": {"filename": "luat_phong_chong_ma_tuy_2021.md"},
                "source": "pageindex",
            }
        ],
    )
    result = module.legal_worker({"question": "Luật ma túy", "top_k": 2})
    assert result["legal_results"][0]["metadata"]["type"] == "legal"
    assert (
        result["legal_results"][0]["metadata"]["source"]
        == "luat_phong_chong_ma_tuy_2021.md"
    )


def test_evidence_worker_deduplicates_and_keeps_highest_score():
    from src.supervisor_workers import evidence_worker

    state = {
        "legal_results": [
            _chunk("Same evidence", 0.5, "legal", "shared.md"),
            _chunk("Legal only", 0.7, "legal", "legal.md"),
        ],
        "news_results": [
            _chunk("Same evidence", 0.9, "news", "shared.md"),
        ],
    }

    result = evidence_worker(state)

    assert len(result["evidence"]) == 2
    assert result["evidence"][0]["content"] == "Same evidence"
    assert result["evidence"][0]["score"] == 0.9
    assert result["has_sufficient_evidence"] is True


def test_evidence_worker_marks_empty_evidence_as_insufficient():
    from src.supervisor_workers import evidence_worker

    result = evidence_worker({"legal_results": [], "news_results": []})

    assert result == {"evidence": [], "has_sufficient_evidence": False}


def test_graph_dispatches_mixed_query_to_both_retrieval_workers(monkeypatch):
    import src.supervisor_workers as module

    monkeypatch.setattr(
        module,
        "retrieve",
        lambda query, top_k: [
            _chunk("Legal evidence", 0.9, "legal", "law.md"),
            _chunk("News evidence", 0.8, "news", "news.md"),
            _chunk("Lower legal evidence", 0.4, "legal", "law-2.md"),
            _chunk("Lower news evidence", 0.3, "news", "news-2.md"),
        ],
    )

    result = module.run_supervisor_workers(
        "Nghệ sĩ bị bắt sẽ chịu hình phạt nào?",
        top_k=2,
    )

    assert result["route"] == "mixed"
    assert len(result["sources"]) <= 2
    assert {item["content"] for item in result["sources"]} == {
        "Legal evidence",
        "News evidence",
    }
    assert result["worker_errors"] == []
    assert result["answer"]


def test_graph_refuses_to_guess_without_evidence(monkeypatch):
    import src.supervisor_workers as module

    monkeypatch.setattr(module, "retrieve", lambda query, top_k: [])

    result = module.run_supervisor_workers("Quy định pháp luật?", top_k=2)

    assert result["sources"] == []
    assert result["answer"] == "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def test_graph_continues_when_one_worker_fails(monkeypatch):
    import src.supervisor_workers as module

    def failed_legal_worker(state):
        return {"legal_results": [], "worker_errors": ["legal: unavailable"]}

    monkeypatch.setattr(module, "legal_worker", failed_legal_worker)
    monkeypatch.setattr(
        module,
        "retrieve",
        lambda query, top_k: [_chunk("News evidence", 0.8, "news", "news.md")],
    )
    module.build_supervisor_graph.cache_clear()

    result = module.run_supervisor_workers(
        "Nghệ sĩ bị bắt sẽ chịu hình phạt nào?",
        top_k=2,
    )

    assert [item["content"] for item in result["sources"]] == ["News evidence"]
    assert result["worker_errors"] == ["legal: unavailable"]
    assert result["answer"]
    module.build_supervisor_graph.cache_clear()


def test_public_entry_rejects_blank_question_before_retrieval(monkeypatch):
    import src.supervisor_workers as module

    def unexpected_retrieval(query, top_k):
        raise AssertionError("retrieval should not run")

    monkeypatch.setattr(module, "retrieve", unexpected_retrieval)

    with pytest.raises(ValueError, match="không được để trống"):
        module.run_supervisor_workers("   ")


def test_task10_generation_keeps_contract_and_adds_agent_metadata(monkeypatch):
    import src.supervisor_workers as module
    from src.task10_generation import generate_with_citation

    monkeypatch.setattr(
        module,
        "retrieve",
        lambda query, top_k: [_chunk("Legal evidence", 0.9, "legal", "law.md")],
    )

    result = generate_with_citation("Hình phạt theo luật?", top_k=1)

    expected_keys = {"answer", "sources", "retrieval_source", "route", "worker_errors"}
    assert expected_keys <= set(result)
    assert result["route"] == "legal"
