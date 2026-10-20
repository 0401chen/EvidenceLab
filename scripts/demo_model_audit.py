"""Opt-in private acceptance capture. Observes requests; never limits or retries them."""
from __future__ import annotations

from decimal import Decimal
import functools
import json
from pathlib import Path
import threading


def install(app, output: Path) -> None:
    import httpx
    from backend.app import providers

    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if not output.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("Acceptance capture must stay under ignored artifacts/.")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model-usage.json").exists():
        raise ValueError("Use a new capture directory; never overwrite earlier real responses.")
    secret = app.state.config.api_key
    records: list[dict] = []
    stages: list[dict] = []
    lock = threading.RLock()

    def save(name, data):
        text = json.dumps(data, ensure_ascii=False, indent=2)
        if secret:
            text = text.replace(secret, "[REDACTED]")
        (output / name).write_text(text, encoding="utf-8")

    def ledger():
        save("model-usage.json", {"http_requests": len(records), "records": records,
                                 "estimated_usd": str(sum(Decimal(r.get("estimated_usd", "0")) for r in records))})

    ledger()
    original = httpx.AsyncClient.send

    async def observed(client, request, **kwargs):
        # Headers (especially Authorization) and raw URLs are never collected.
        # Only the configured Chat payload is observed; normal transport and
        # compatibility retries remain entirely under the production provider.
        try:
            payload = json.loads(request.content)
        except (ValueError, UnicodeError):
            return await original(client, request, **kwargs)
        if not isinstance(payload, dict) or "messages" not in payload or "model" not in payload:
            return await original(client, request, **kwargs)
        with lock:
            number = len(records) + 1
            record = {"number": number, "model": payload["model"], "status": "sent"}
            records.append(record)
            save(f"request-{number}.json", payload)
            ledger()
        try:
            response = await original(client, request, **kwargs)
            await response.aread()
            try:
                body = response.json()
            except ValueError:
                body = {"non_json": True}
            with lock:
                record.update(status="returned", http_status=response.status_code)
                usage = body.get("usage") if isinstance(body, dict) else None
                if isinstance(usage, dict):
                    record["usage"] = usage
                    # User-confirmed rates for this configured demonstration
                    # model only. Unknown models have usage, not invented cost.
                    if (record["model"] == "gpt-4o-mini"
                            and type(usage.get("prompt_tokens")) is int
                            and type(usage.get("completion_tokens")) is int):
                        record["estimated_usd"] = str((Decimal(usage["prompt_tokens"]) * Decimal("0.15")
                                                       + Decimal(usage["completion_tokens"]) * Decimal("0.60")) / 1000000)
                save(f"response-{number}.json", body)
            return response
        except BaseException as exc:
            record.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            with lock:
                ledger()

    httpx.AsyncClient.send = observed

    def capture(function, label):
        @functools.wraps(function)
        async def wrapped(*args, **kwargs):
            result = await function(*args, **kwargs)
            with lock:
                stages.append({"stage": label, "result": result})
                save("stages.json", stages)
            return result
        return wrapped

    providers.answer_llm = capture(providers.answer_llm, "answer")
    providers.verify_claim = capture(providers.verify_claim, "verification")

    results = []

    @app.middleware("http")
    async def capture_result(request, call_next):
        if request.method != "POST" or request.url.path not in ("/api/qa", "/api/verify"):
            return await call_next(request)
        body = await request.json()
        response = await call_next(request)
        chunks = [part async for part in response.body_iterator]
        content = b"".join(part if isinstance(part, bytes) else part.encode() for part in chunks)
        try:
            result = json.loads(content)
        except ValueError:
            result = {"non_json": True}
        with lock:
            results.append({"path": request.url.path, "input": body, "http_status": response.status_code,
                            "result": result})
            save("api-results.json", results)
        from starlette.responses import Response
        return Response(content=content, status_code=response.status_code,
                        headers=dict(response.headers), media_type=response.media_type)
