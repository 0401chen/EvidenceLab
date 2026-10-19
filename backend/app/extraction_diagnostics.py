"""Explicit local observability only. Does not repair, retry or accept results."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import copy
import json
from pathlib import Path
import threading
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[2]
REQUEST_OPTIONS: ContextVar[dict | None] = ContextVar("extraction_request_options", default=None)
_ACTIVE: ContextVar["LocalCapture | None"] = ContextVar("extraction_capture", default=None)
_FILE_LOCK = threading.Lock()


class LocalCapture:
    """Metadata by default; content requires explicit opt-in. Fail closed on I/O.

    Captures are local, unencrypted and bounded. There is no HTTP download route,
    credential/header serialization, automatic deletion or retention rotation.
    """
    MAX_FILES = 8
    MAX_BYTES = 2_000_000

    def __init__(self, folder: Path, *, include_content=False, secrets=(), workspace=ROOT):
        self.path = None
        self.enabled = False
        self.truncated = False
        self.bytes = 0
        self.include_content = include_content
        self.secrets = tuple(s for s in secrets if isinstance(s, str) and len(s) >= 8)
        self.stream = None
        try:
            folder = Path(folder).resolve()
            if not folder.is_relative_to((Path(workspace).resolve()/"artifacts").resolve()):
                return
            # Also reject an artifacts junction which resolves outside workspace.
            if not folder.is_relative_to(Path(workspace).resolve()):
                return
            with _FILE_LOCK:
                folder.mkdir(parents=True, exist_ok=True)
                if len(list(folder.glob("*.jsonl"))) >= self.MAX_FILES:
                    return
                self.path = folder/(uuid.uuid4().hex+".jsonl")
                self.stream = self.path.open("x", encoding="utf8")
                self.enabled = True
            self.record("capture_started", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "content_collected": bool(include_content), "schema": "extraction_diagnostic_v1"})
        except (OSError, ValueError):
            self.enabled = False

    def _redact(self, value):
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {self._redact(str(k)): "[REDACTED]" if str(k).lower() in {
                "authorization", "api_key", "password", "access_token"}
                else self._redact(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._redact(v) for v in value]
        return value

    def record(self, event: str, data: dict):
        if not self.enabled:
            return
        try:
            redacted = self._redact(data)
            encoded = json.dumps(redacted, ensure_ascii=False, allow_nan=False)
            # These keys contain server-controlled categories/options, not prose.
            meta = {k: redacted[k] for k in ("created_at_utc", "issues", "aliases", "request_options", "model", "document",
                    "provider_source_sha256", "provider_code", "http_status",
                    "checked_records_returned", "extraction_records_returned", "capture_complete", "content_collected", "schema") if k in redacted}
            item = {"event": event, "data_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                    "redacted": redacted != data, **meta}
            if self.include_content:
                item["data"] = redacted
            line = json.dumps(item, ensure_ascii=False, allow_nan=False)+"\n"
            size = len(line.encode())
            if self.bytes+size > self.MAX_BYTES-4096 and event != "capture_finished":
                self.truncated = True
                return
            if self.bytes+size > self.MAX_BYTES:
                self.truncated = True
                return
            self.stream.write(line)
            self.stream.flush()
            self.bytes += size
        except Exception:
            # Observability must never change the extraction result or error.
            self.truncated = True
            self.enabled = False

    def finish(self, provider_code=None):
        self.record("capture_finished", {"capture_complete": self.enabled and not self.truncated,
                                          "provider_code": provider_code})
        if self.stream is not None:
            try:
                self.stream.close()
            except OSError:
                pass


@contextmanager
def diagnostic_session(folder: Path, *, include_content=False, workspace=ROOT, document_metadata=None,
                       require_complete=False):
    """Install hooks only for an explicitly launched local diagnostic process.

    Context variables isolate simultaneous operations. Every existing model call
    and validation runs exactly once as before; no output is reassigned or fixed.
    """
    import httpx
    from . import providers

    original_extract = providers.extract_llm
    original_check = providers._checked_extraction
    original_screen = providers._screen_extraction
    original_enrich = providers._enrich_extraction_batch
    original_send = httpx.AsyncClient.send

    async def extract(evidence, paper_id, config):
        capture = LocalCapture(folder, include_content=include_content, workspace=workspace,
                               secrets=(config.get("api_key"),))
        token = _ACTIVE.set(capture)
        failure = None
        try:
            if require_complete and (not capture.enabled or not include_content):
                raise RuntimeError("diagnostic_capture_unavailable")
            try:
                source_hash = hashlib.sha256(Path(providers.__file__).read_bytes()).hexdigest()
            except OSError:
                source_hash = None
            try:
                document = document_metadata(paper_id) if document_metadata else None
            except Exception:
                document = {"metadata_unavailable": True}
            capture.record("extraction_input", {"paper_id": paper_id, "evidence": evidence,
                "request_options": REQUEST_OPTIONS.get() or {"max_results": config.get("max_results", 12)},
                "provider_source_sha256": source_hash,
                "document": document, "model": config.get("model")})
            records = await original_extract(evidence, paper_id, config)
            capture.record("extraction_result", {"extraction_records_returned": len(records), "records": records})
            return records
        except Exception as exc:
            failure = exc.code if isinstance(exc, providers.ProviderError) else "unexpected_error"
            raise
        finally:
            capture.finish(failure)
            _ACTIVE.reset(token)

    def checked(result, batch, paper_id, aliases, offset, **kwargs):
        capture = _ACTIVE.get()
        if capture is None:
            return original_check(result, batch, paper_id, aliases, offset, **kwargs)
        issues = kwargs.get("diagnostics")
        if issues is None:
            issues = []
            kwargs["diagnostics"] = issues
        outcome, failure = 0, None
        try:
            records = original_check(result, batch, paper_id, aliases, offset, **kwargs)
            outcome = len(records)
            return records
        except Exception as exc:
            failure = exc.code if isinstance(exc, providers.ProviderError) else "unexpected_error"
            raise
        finally:
            capture.record("validation", {"aliases": aliases, "issues": issues, "checked_records_returned": outcome,
                "provider_code": failure, "result": result, "original_batch": batch, "offset": offset})

    def screen(result, batch, paper_id, aliases):
        checked_rows, rejected = original_screen(result, batch, paper_id, aliases)
        capture = _ACTIVE.get()
        if capture is not None:
            capture.record("screen_result", {"input_count": len(result.get("experiments", [])),
                "accepted_count": len(checked_rows), "rejected_count": len(rejected),
                "rejections": rejected, "records": checked_rows})
        return checked_rows, rejected

    async def enrich(experiments, scope, config):
        capture = _ACTIVE.get()
        before = copy.deepcopy(experiments) if capture is not None else None
        try:
            return await original_enrich(experiments, scope, config)
        finally:
            if capture is not None:
                capture.record("enrichment_result", {"before": before, "after": experiments})

    async def send(client, request, *args, **kwargs):
        capture = _ACTIVE.get()
        selected = capture is not None and request.method == "POST" and request.url.path.rstrip("/").endswith("/chat/completions")
        if selected:
            try:
                payload = json.loads(request.content)
                capture.record("model_request", {k: payload[k] for k in (
                    "model", "messages", "temperature", "max_tokens", "max_completion_tokens", "response_format") if k in payload})
            except (ValueError, TypeError):
                capture.record("model_request_unavailable", {})
            if require_complete and (not capture.enabled or capture.truncated):
                raise RuntimeError("diagnostic_capture_unavailable")
        response = await original_send(client, request, *args, **kwargs)
        if selected:
            try:
                # Preserve raw message content before the provider parses JSON.
                # Error bodies, HTTP headers and endpoint credentials are excluded.
                body = response.json() if response.status_code == 200 else {}
                capture.record("model_response", {"http_status": response.status_code, "choices": body.get("choices", []),
                                                   "usage": body.get("usage")})
            except (ValueError, TypeError, AttributeError):
                capture.record("model_response_unavailable", {"http_status": response.status_code})
        return response

    with patch.object(providers, "extract_llm", extract), patch.object(providers, "_checked_extraction", checked), \
            patch.object(providers, "_screen_extraction", screen), patch.object(providers, "_enrich_extraction_batch", enrich), \
            patch.object(httpx.AsyncClient, "send", send):
        yield
