"""Isolated, offline test fixtures for the local evidence workbench."""

from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Delayed import keeps test collection free of persistent application state.
    monkeypatch.setenv("EVIDENCELAB_DATA_DIR", str(tmp_path / "data"))
    # Explicit empty values also prevent python-dotenv loading real credentials.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("LLM_BASE_URL", "http://offline.invalid/v1")
    monkeypatch.setenv("EVIDENCE_PARSER", "pymupdf")
    monkeypatch.setenv("LLM_TIMEOUT", "90")
    from backend.app.main import create_app

    with TestClient(create_app(tmp_path / "data")) as test_client:
        yield test_client


@pytest.fixture
def evidence_factory():
    def make(text, evidence_id="ev-a", paper_id="paper-a", page=1):
        return {
            "id": evidence_id,
            "paper_id": paper_id,
            "page": page,
            "section": "Results",
            "text": text,
            "bbox": None,
            "source_type": "text",
        }

    return make


@pytest.fixture
def experiment_factory():
    def make(experiment_id="exp-a", paper_id="paper-a", **overrides):
        result = {
            "id": experiment_id,
            "paper_id": paper_id,
            "name": experiment_id,
            "task": "classification",
            "method": "AlphaNet",
            "dataset": "DemoSet",
            "split": "test-A",
            "metric": "accuracy",
            "value": 90.0,
            "unit": "%",
            "sample_size": "100",
            "conditions": {"temperature": "25 C", "protocol": "fixed"},
            "evidence_ids": [f"ev-{experiment_id}"],
            "extraction_method": "rule",
            "notes": [],
            "field_evidence": {"value": [f"ev-{experiment_id}"]},
            "evidence_quotes": {"value": [{"id": f"ev-{experiment_id}", "quote": f"Value: {overrides.get('value', 90.0):g}"}]},
        }
        result.update(overrides)
        return result

    return make
