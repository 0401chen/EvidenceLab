"""Default selection and opt-in online preview; never send real model traffic."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

import httpx
import pytest

from backend.app.config import Config
from scripts import serve_preview_ai as preview


@pytest.fixture
def clean_config(monkeypatch, tmp_path):
    for name in ("LLM_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_MODEL",
                 "LLM_ALLOWED_MODELS", "LLM_TIMEOUT", "PYTHON_DOTENV_DISABLED", "EVIDENCELAB_DATA_DIR", "EVIDENCE_PARSER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("EVIDENCELAB_DATA_DIR", str(tmp_path / "unrelated-import"))
    return monkeypatch


@pytest.mark.parametrize("value,expected", [(None, "gpt-4o-mini"), ("", "gpt-4o-mini"),
    ("   ", "gpt-4o-mini"), ("gpt-4.1-nano", "gpt-4.1-nano")])
def test_default_and_explicit_private_model_are_consistent(clean_config, tmp_path, value, expected):
    clean_config.setenv("LLM_API_KEY", "synthetic-private-test-key")
    clean_config.setenv("LLM_BASE_URL", "https://provider.invalid/v1")
    if value is not None:
        clean_config.setenv("LLM_MODEL", value)
    cfg = Config.from_env(tmp_path)
    assert cfg.model == cfg.provider_config()["model"] == expected
    assert cfg.llm_configured and cfg.public()["qa_engine"] == "llm"
    assert cfg.public()["provider_status"]["chat"]["status"] == "untested"


def test_no_key_shows_default_without_claiming_configuration(clean_config, tmp_path):
    cfg = Config.from_env(tmp_path)
    public = cfg.public()
    assert cfg.model == "gpt-4o-mini" and not cfg.llm_configured
    assert public["qa_engine"] == "local"
    assert not any(m["enabled"] for m in public["models"])
    assert "尚未连接模型服务" in public["provider_status"]["chat"]["message"]


@pytest.mark.parametrize("endpoint", ["", "bad-address", "https://user:private@provider.invalid/v1",
    "https://provider.invalid/v1?token=private", "https://provider.invalid:bad/v1"])
def test_invalid_provider_is_not_presented_as_configured(clean_config, tmp_path, endpoint):
    clean_config.setenv("LLM_API_KEY", "synthetic-private-test-key")
    clean_config.setenv("LLM_BASE_URL", endpoint)
    cfg = Config.from_env(tmp_path)
    assert not cfg.llm_configured and cfg.public()["qa_engine"] == "local"


def test_default_permission_failure_and_explicit_switch_never_fallback(client, monkeypatch):
    cfg = client.app.state.config
    object.__setattr__(cfg, "api_key", "synthetic-private-test-key")
    object.__setattr__(cfg, "allowed_models", ("gpt-4.1-mini", "gpt-4.1-nano"))
    assert cfg.model == "gpt-4o-mini"
    calls = []
    original = httpx.AsyncClient
    def handler(request):
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(404, json={"error": "model not enabled"})
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    pid = client.post("/api/papers/upload", files={"files": ("source.txt", b"Accuracy was 81.5%.", "text/plain")}).json()["papers"][0]["id"]
    for _ in range(2):
        assert client.get("/api/health").status_code == client.get("/api/settings").status_code == 200
    assert calls == []
    for model in [None, "gpt-4.1-mini", "gpt-4.1-nano"]:
        response = client.post("/api/qa", json={"paper_ids": [pid], "question": "accuracy?", "engine": "llm",
                                              **({"model_id": model} if model else {})})
        assert response.status_code == 502 and "未切换模型" in response.json()["detail"]
    assert calls == ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"]


@contextlib.contextmanager
def preview_case():
    base = preview.ROOT / "artifacts/previews"
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="preview-contract-", dir=base) as directory:
        root = Path(directory).resolve()
        assert root.is_relative_to(base.resolve())
        frontend = root / "frontend"
        frontend.mkdir()
        (frontend / "index.html").write_text("<html>Synthetic preview</html>", encoding="utf8")
        config = root / "private.env"
        config.write_text("LLM_API_KEY=synthetic-private-test-key\nLLM_BASE_URL=https://provider.invalid/v1\n"
                          "LLM_MODEL=\nEVIDENCELAB_DATA_DIR=D:/must-not-use-formal-data\n", encoding="utf8")
        yield root, frontend, config


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_preview_check_is_read_only_and_hides_private_values(clean_config, capsys):
    with preview_case() as (root, frontend, config):
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        library = root / "new-library"
        preview.main(["--check", "--port", str(free_port()), "--data-dir", str(library),
                      "--frontend-dir", str(frontend), "--config-file", str(config),
                      "--allowed-models", "gpt-4.1-mini,gpt-4.1-nano"])
        captured = capsys.readouterr().out
        result = json.loads(captured)
        assert result["configured"] and result["model"] == "gpt-4o-mini"
        assert result["startup_model_requests"] == 0 and result["connection"] == "untested"
        assert result["data_dir"] == str(library.resolve())
        assert "synthetic-private-test-key" not in captured
        assert not library.exists() and hashlib.sha256(config.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("port", [8000, 8790, 8792, 8793])
def test_preview_cannot_use_protected_service_ports(clean_config, port):
    with pytest.raises(SystemExit) as error:
        preview.main(["--port", str(port)])
    assert error.value.code == 2


def test_occupied_port_fails_before_creating_library(clean_config, capsys):
    with preview_case() as (root, frontend, config), socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0)); occupied.listen()
        library = root / "new-library"
        with pytest.raises(SystemExit):
            preview.main(["--port", str(occupied.getsockname()[1]), "--data-dir", str(library),
                          "--frontend-dir", str(frontend), "--config-file", str(config)])
        assert not library.exists()
        assert "occupied" in capsys.readouterr().err


def test_preview_never_accepts_formal_or_unowned_existing_data(clean_config):
    with pytest.raises(preview.PreviewError):
        preview.preview_directory(preview.ROOT / "artifacts/demo-data")
    with preview_case() as (root, _, _):
        library = root / "unowned"
        library.mkdir()
        (library / "evidencelab.sqlite3").write_bytes(b"unrelated existing data")
        with pytest.raises(preview.PreviewError):
            preview.preview_directory(library)


def test_online_launcher_uses_isolated_db_and_normal_mocked_transport(clean_config):
    with preview_case() as (root, frontend, config):
        data = root / "online-library"
        code = r'''
import asyncio, json, runpy, sys
from pathlib import Path
import uvicorn, httpx
class PreviewServer:
    def __init__(self, config): self.app=config.app
    def run(self, sockets):
        cfg=self.app.state.config
        assert cfg.data_dir==Path(sys.argv[sys.argv.index('--data-dir')+1]).resolve()
        assert (cfg.data_dir/'evidencelab.sqlite3').is_file()
        assert len(self.app.state.store.list_papers())==0
        from backend.app import providers
        assert providers._chat_json.__name__=='_chat_json'
        calls=[]
        def handler(request):
            calls.append(json.loads(request.content)['model'])
            return httpx.Response(200,json={'model':'gpt-4o-mini','choices':[{'message':{'content':'{"ok":true}'}}]})
        original=httpx.AsyncClient
        httpx.AsyncClient=lambda **kw: original(transport=httpx.MockTransport(handler),**kw)
        assert calls==[]
        assert asyncio.run(providers._chat_json([],cfg.provider_config()))=={'ok':True}
        assert calls==['gpt-4o-mini']
uvicorn.Server=PreviewServer
runpy.run_path('scripts/serve_preview_ai.py',run_name='__main__')
'''
        result = subprocess.run([sys.executable, "-B", "-c", code, "--port", str(free_port()),
            "--data-dir", str(data), "--frontend-dir", str(frontend), "--config-file", str(config)],
            cwd=preview.ROOT, capture_output=True, text=True, env=os.environ.copy())
        assert result.returncode == 0, "Isolated online preview/transport contract failed"
        assert "synthetic-private-test-key" not in result.stdout + result.stderr
        assert (data / ".preview-owner.json").is_file()
        assert preview.preview_directory(data) == data.resolve()  # Owned preview is safely reusable.
