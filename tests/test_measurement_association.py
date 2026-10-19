"""A literal metric quotation must refer to this measurement, not its neighbor."""
import asyncio

import pytest

from backend.app import providers
from backend.app.errors import ProviderError


def supplied(value, quote):
    return {"value": value, "evidence": [{"id": "S1", "quote": quote}]}


@pytest.mark.parametrize("unit,metric,quote", [
    ("%", "accuracy", "accuracy of 84%"),
    (None, "accuracy", "accuracy was 84"),
    ("ms", "latency", "latency of 84 ms"),
])
def test_metric_quote_with_a_different_measurement_is_rejected(evidence_factory, unit, metric, quote):
    measurement = f"{metric} of 82" + (f" {unit}" if unit else "")
    text = f"Prior system reports {quote}. Our system reports {measurement}."
    source = evidence_factory(text, "source")
    row = {"metric": supplied(metric, quote), "value": supplied(82, measurement)}
    if unit:
        row["unit"] = supplied(unit, measurement)
    with pytest.raises(ProviderError, match="metric"):
        providers._checked_extraction({"experiments": [row]}, [source], "paper-a", {"S1": "source"}, 0)


def test_metric_value_mismatch_uses_bounded_correction_not_silent_reassignment(monkeypatch, evidence_factory):
    source = evidence_factory("Prior method has accuracy of 84%. Our method has error rate of 82%.", "source")
    calls = []

    async def reply(messages, config):
        calls.append(messages)
        row = {"metric": supplied("accuracy", "accuracy of 84%"),
               "value": supplied(82, "error rate of 82%"), "unit": supplied("%", "error rate of 82%")}
        if len(calls) == 2:
            assert "metric" in messages[1]["content"]
            row["metric"] = supplied("error rate", "error rate of 82%")
        return {"experiments": [row]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    rows = asyncio.run(providers.extract_llm([source], "paper-a", {"model": "offline", "enrich_metadata": False}))
    assert len(calls) == 2
    assert len(rows) == 1 and rows[0]["metric"] == "error rate" and rows[0]["value"] == 82
    assert rows[0]["method"] is None
