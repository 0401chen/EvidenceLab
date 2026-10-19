"""Scientific vetoes must preserve uncertainty, not invent corrected facts."""
import asyncio

import pytest

from backend.app import providers
from backend.app.research_guard import numeric_claim_gaps


@pytest.mark.parametrize("claim,source,verdict", [
    ("AlphaNet-3 在 BenchSet-C 的错误率为12%。", "AlphaNet-3 error on BenchSet is 12%.", "contradicted"),
    ("AlphaNet-3 在 BenchSet 的错误率为12%。", "AlphaNet-3 accuracy on BenchSet is 12%.", "supported"),
    ("AlphaNet-3 top-1错误率为12%。", "AlphaNet-3 12% 3%", "supported"),
    ("AlphaNet-3 单裁剪 top-1 错误率为12%。", "AlphaNet-3 top-1 error 12 (10), top-5 error 3 (2); single-crop (ten-crop).", "supported"),
    ("AlphaNet-3 在验证集的准确率为88%。", "AlphaNet-3 accuracy is 88%.", "supported"),
    ("AlphaNet-3准确率为89%。", "AlphaNet-3 accuracy is 88%.", "supported"),
    ("AlphaNet-3 在 Corpus-X 准确率为88%。", "AlphaNet-3 accuracy on CorpusXtra is 88%.", "supported"),
    ("AlphaNet-3 英法翻译为30 BLEU。", "AlphaNet-3 English-to-German achieved 32 BLEU.", "contradicted"),
    ("AlphaNet-3错误率为88%。", "AlphaNet-3 accuracy is 88%, and error is 12%.", "supported"),
    ("基础 AlphaNet 达到每秒128帧。", "Our base AlphaNet runs at 36 frames per second. A smaller AlphaNet runs at 128 frames per second.", "supported"),
    ("模型准确率为百分之八十五。", "The model error rate is 85%.", "supported"),
])
def test_unresolved_scope_direction_column_protocol_or_value_abstains(claim, source, verdict):
    assert numeric_claim_gaps(claim, [{"quote": source}], verdict)


@pytest.mark.parametrize("claim,source,verdict", [
    ("AlphaNet-3 准确率为88%。", "AlphaNet-3 accuracy is 88%.", "supported"),
    ("AlphaNet-3 在 task-B 上达到30 BLEU。", "AlphaNet-3 on task-B achieved 32 BLEU, not 30.", "contradicted"),
    ("未报告运行时间。", "AlphaNet-3 score is 20.", "insufficient"),
    ("AlphaNet-3 的 top-1 准确率为88%。", "AlphaNet-3 top-1 accuracy is 88%.", "supported"),
    ("AlphaNet-3 在验证集准确率为88%。", "AlphaNet-3 在验证集准确率达到88%。", "supported"),
    ("AlphaNet-3 准确率为88%。", "AlphaNet-3 accuracy is 88%; we also report Corpus-100 (91.7%) elsewhere.", "supported"),
    ("AlphaNet-3 英法翻译达到30 BLEU。", "AlphaNet-3 English-to-French achieved 32 BLEU.", "contradicted"),
    ("基础 AlphaNet 达到每秒36帧。", "Our base AlphaNet runs at 36fps. A smaller AlphaNet runs at 128fps.", "supported"),
    ("基础 AlphaNet 达到每秒128帧。", "Our base AlphaNet runs at 36 frames per second, not 128.", "contradicted"),
    ("模型达到28.4 BLEU。", "The model reached 28.4BLEU.", "supported"),
])
def test_explicit_support_and_different_value_refutation_remain_possible(claim, source, verdict):
    assert not numeric_claim_gaps(claim, [{"quote": source}], verdict)


def test_wrong_numeric_answer_is_withheld_without_protocol_retry(monkeypatch, evidence_factory):
    calls = []

    async def reply(messages, config):
        calls.append(messages)
        return {"assertions": [{"text": "AlphaNet-3 的错误率为88%。", "evidence": [{"handle": "F1"}]}]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    result = asyncio.run(providers.answer_llm("AlphaNet-3 error?", [evidence_factory("AlphaNet-3 accuracy is 88%.")], {"model": "offline", "task_protocol": "legacy"}))
    assert len(calls) == 1  # Semantic uncertainty must not trigger a paid format correction.
    assert result["verdict"] == "insufficient"
    assert "错误率为88" not in result["answer"]
    assert result["withheld_assertions"][0]["text"].endswith("错误率为88%。")
    assert result["citation_quotes"][0]["quote"] == "AlphaNet-3 accuracy is 88%."


def test_unsafe_model_verdict_is_auditable_and_not_rewritten_to_opposite(monkeypatch, evidence_factory):
    claim = "AlphaNet-3 在 BenchSet-C 的错误率为12%。"

    async def reply(messages, config):
        return {"checks": [{"claim": claim, "verdict": "contradicted", "explanation": "另一个数据集是10%。",
                            "evidence": [{"handle": "F1"}]}]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    result = asyncio.run(providers.verify_claim(claim, [evidence_factory("AlphaNet-3 error on BenchSet is 10%.")], {"model": "offline", "task_protocol": "legacy"}))
    assert result["verdict"] == "insufficient"
    check = result["claim_checks"][0]
    assert check["model_verdict"] == "contradicted"
    assert check["safety_gaps"]
    assert "确定性来源检查未通过" in result["explanation"]


def test_api_does_not_semantically_reverify_a_withheld_answer(client, monkeypatch):
    from backend.app.config import Config
    monkeypatch.setattr(Config, "provider_config", lambda self: {"model": "offline"})
    monkeypatch.setattr(Config, "llm_configured", property(lambda self: True))
    # Reuse normal upload validation and do not touch the formal research DB.
    import fitz
    document = fitz.open()
    document.new_page().insert_text((72, 72), "AlphaNet-3 accuracy is 88%.")
    paper = client.post("/api/papers/upload", files=[("files", ("guard.pdf", document.tobytes(), "application/pdf"))]).json()["papers"][0]
    document.close()
    calls = []
    sources = client.get(f"/api/papers/{paper['id']}").json()["evidence"]

    async def answer(*args, **kwargs):
        calls.append("answer")
        return {"answer": "证据不足", "verdict": "insufficient", "mode": "llm", "citations": sources, "warnings": []}

    async def verify(*args, **kwargs):
        pytest.fail("An abstained answer must not trigger a second model request")

    monkeypatch.setattr(providers, "answer_llm", answer)
    monkeypatch.setattr(providers, "verify_claim", verify)
    response = client.post("/api/qa", json={"paper_ids": [paper["id"]], "question": "accuracy?", "engine": "llm"})
    assert response.status_code == 200
    assert calls == ["answer"]
