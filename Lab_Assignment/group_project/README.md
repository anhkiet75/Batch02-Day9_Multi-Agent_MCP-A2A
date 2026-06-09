# Bài Tập Nhóm — Search Engine / RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1:  Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

### Code mẫu — DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Tạo test cases từ golden dataset
test_cases = []
for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=[c["content"] for c in result["sources"]],
    )
    test_cases.append(test_case)

# Chạy evaluation
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
]

results = evaluate(test_cases, metrics)
```

### Code mẫu — RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Chuẩn bị data
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append([c["content"] for c in result["sources"]])
    eval_data["ground_truth"].append(item["expected_answer"])

dataset = Dataset.from_dict(eval_data)

# Chạy evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result.to_pandas())
```

### Code mẫu — TruLens

```python
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

# Define feedback functions
f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
f_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = Feedback(provider.context_relevance).on_input()

# Wrap RAG pipeline
tru_rag = TruCustomApp(
    rag_pipeline,
    app_name="DrugLaw_RAG",
    feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
)

# Run evaluation
with tru_rag as recording:
    for item in golden_dataset:
        rag_pipeline.generate_with_citation(item["question"])

# View dashboard
from trulens.dashboard import run_dashboard
run_dashboard()
```

### Deliverable Evaluation

- [x] File `group_project/evaluation/golden_dataset.json` — 16 cặp Q&A
- [x] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [x] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [x] So sánh A/B ít nhất 2 configs (hybrid_rerank vs dense_only)

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
[Input Query]
      │
      ▼
┌─────────────────────────────────────────────┐
│              Retrieval Pipeline              │
│                                             │
│  ┌──────────────┐   ┌──────────────────┐   │
│  │ Semantic Search│   │  Lexical Search  │   │
│  │ (Weaviate    │   │  (BM25 / task6)  │   │
│  │  task5)      │   │                  │   │
│  └──────┬───────┘   └────────┬─────────┘   │
│         │                    │             │
│         └──────────┬─────────┘             │
│                    ▼                       │
│             RRF Fusion (task7)             │
│                    │                       │
│                    ▼                       │
│          Cross-Encoder Reranking           │
│                 (task7)                    │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│           Generation (task10)               │
│     GPT-4o-mini + Citation Prompt           │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
               [Answer + Sources]
```

---

## Kết Quả Evaluation

Framework: **DeepEval** | Model: `gpt-4o-mini` | Chi tiết: [`evaluation/results.md`](evaluation/results.md)

### A/B Comparison (8 cases — cả 2 configs hoàn thành)

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|-----------------------------|-----------------------|---|
| Faithfulness | 0.5833 | 1.0000 | -0.4167 |
| Answer Relevance | 0.9821 | 0.0000 | +0.9821 |
| Context Recall | 1.0000 | 0.1250 | +0.8750 |
| Context Precision | 0.7361 | 0.0000 | +0.7361 |
| **Average** | **0.8254** | **0.2812** | **+0.5442** |

### Full Dataset (16 cases — dense-only)

| Metric | Config B (dense-only) |
|--------|-----------------------|
| Faithfulness | 1.0000 |
| Answer Relevance | 0.1250 |
| Context Recall | 0.1042 |
| Context Precision | 0.0000 |
| **Average** | **0.3073** |

**Kết luận:** Config A (hybrid + rerank) vượt trội Config B trên 3/4 metrics với điểm trung bình 0.8254 vs 0.2812. Dense-only đạt Faithfulness cao nhưng Answer Relevance và Context Recall thấp do embedding fallback không đủ mạnh.

---

## Phân Công Công Việc

| STT | Họ Tên | MSSV | GitHub | Nhiệm vụ | Trạng thái |
|-----|--------|------|--------|----------|------------|
| 1 | Nguyễn Quang Hoà | 2A202600986 | [nguyenquanghoa1981](https://github.com/nguyenquanghoa1981) | Tạo Golden Dataset — câu hỏi pháp luật ma tuý | ✅ Hoàn thành |
| 2 | Tiền Anh Kiệt | 2A202600961 | [anhkiet75](https://github.com/anhkiet75) | Xây dựng Evaluation Pipeline, cấu hình A/B (hybrid vs dense-only) | ✅ Hoàn thành |
| 3 | Nguyễn Văn Phúc | 2A202600539 | [phucn01](https://github.com/phucn01) | Chạy evaluation, phân tích kết quả & Worst Performers | ✅ Hoàn thành |
| 4 | Nguyễn Hoàng Dương | 2A202600849 | [alihduong](https://github.com/alihduong) | Tích hợp Retrieval Pipeline (hybrid + reranking) vào eval | ✅ Hoàn thành |
| 5 | Vũ Đình Phượng | 2A202600634 | [royalpoinciana234](https://github.com/royalpoinciana234) | Báo cáo kết quả, đề xuất cải tiến, README | ✅ Hoàn thành |

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy evaluation pipeline
VECTOR_STORE=weaviate DEEPEVAL_CONFIG_MODE=ab python3 group_project/evaluation/eval_pipeline.py

# Chạy app
streamlit run app.py
# hoặc
chainlit run app.py
```

---

## Lưu ý: Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
