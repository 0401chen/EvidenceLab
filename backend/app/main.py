from __future__ import annotations

import hashlib
import asyncio
import copy
import json
import logging
import math
import re
import time
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from functools import wraps
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .config import ROOT, Config
from .errors import DocumentError, ProviderError
from .experiment_validation import (prepare_experiments, verify_pdf_measurements,
                                    machine_measurement_verified, prioritize_measurements)
from .store import Store
from .task_attribution import machine_task_verified

VERSION = "0.1.0"
MAX_FILE_BYTES = 32 * 1024 * 1024
logger = logging.getLogger("evidencelab")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ScopeRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list, max_length=30)


class QARequest(ScopeRequest):
    question: str = Field(min_length=1, max_length=4000)
    engine: Literal["local", "llm"] = "local"
    model_id: str | None = Field(default=None, min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")

    @field_validator("question")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("问题不能为空")
        return value.strip()


class VerifyRequest(ScopeRequest):
    claim: str = Field(min_length=1, max_length=4000)
    engine: Literal["local", "llm"] = "local"
    model_id: str | None = Field(default=None, min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")

    @field_validator("claim")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("待核验结论不能为空")
        return value.strip()


class ExtractionRequest(BaseModel):
    mode: Literal["rule", "llm"] = "rule"
    model_id: str | None = Field(default=None, min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
    reparse: bool = False
    pages: list[int] | None = Field(default=None, min_length=1, max_length=25)
    max_results: int = Field(default=12, ge=1, le=30, strict=True)

    @field_validator("pages", mode="before")
    @classmethod
    def valid_pages(cls, value):
        if value is None:
            return None
        if not isinstance(value, list) or any(type(page) is not int or page < 1 for page in value):
            raise ValueError("页码必须是从 1 开始的整数列表")
        return sorted(set(value))


class CandidateAcceptance(BaseModel):
    experiment_ids: list[str] | None = Field(default=None, max_length=500)


class ManualQuote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=200)
    quote: str = Field(min_length=1, max_length=20000)


class ManualCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=180)
    fields: dict[str, str | float | dict[str, str] | None] = Field(max_length=9)
    evidence_quotes: dict[str, list[ManualQuote]] = Field(max_length=9)

    @field_validator("fields", mode="before")
    @classmethod
    def bounded_fields(cls, fields):
        if not isinstance(fields, dict):
            raise ValueError("实验字段必须是对象")
        for key, value in fields.items():
            if key == "conditions":
                if value is not None and (not isinstance(value, dict) or len(value) > 40
                        or any(not isinstance(k, str) or len(k) > 200 or not isinstance(v, str)
                               or len(v) > 1000 for k, v in value.items())):
                    raise ValueError("实验条件格式或长度无效")
            elif key == "value":
                if isinstance(value, bool) or isinstance(value, (list, dict)):
                    raise ValueError("结果数值格式无效")
            elif value is not None and (not isinstance(value, str) or len(value) > 500):
                raise ValueError("实验字段格式或长度无效")
        return fields

    @field_validator("evidence_quotes")
    @classmethod
    def bounded_references(cls, quotes):
        if any(len(refs) > 12 for refs in quotes.values()):
            raise ValueError("每个字段最多关联 12 条原文")
        return quotes


class ExperimentPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    name: str | None = Field(default=None, max_length=500)
    task: str | None = Field(default=None, max_length=500)
    method: str | None = Field(default=None, max_length=500)
    dataset: str | None = Field(default=None, max_length=500)
    split: str | None = Field(default=None, max_length=500)
    metric: str | None = Field(default=None, max_length=200)
    value: float | None = None
    unit: str | None = Field(default=None, max_length=100)
    sample_size: str | None = Field(default=None, max_length=200)
    conditions: dict[str, str] | None = None
    evidence_ids: list[str] | None = None
    field_evidence: dict[str, list[str]] | None = None

    @field_validator("value", mode="before")
    @classmethod
    def finite(cls, value):
        if isinstance(value, bool):
            raise ValueError("实验数值不能是布尔值")
        try:
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("实验数值必须是有限数")
        except (TypeError, OverflowError):
            raise ValueError("实验数值格式无效") from None
        return value

    @field_validator("name")
    @classmethod
    def valid_name(cls, value):
        if value is None or not value.strip():
            raise ValueError("实验名称不能为空")
        return value.strip()


def create_app(data_dir: str | Path | None = None) -> FastAPI:
    cfg = Config.from_env(data_dir)
    store = Store(cfg.data_dir)
    app = FastAPI(title="EvidenceLab · 科研文献证据分析与结论核验", version=VERSION)
    app.state.store = store
    app.state.config = cfg
    app.state.provider_status = {}
    inflight: set[str] = set()
    inflight_lock = threading.Lock()
    preview_slots = asyncio.Semaphore(2)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000", "http://127.0.0.1:8000"], allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["Content-Type"])

    def paper_or_404(paper_id):
        item = store.get_paper(paper_id)
        if item is None:
            raise HTTPException(404, "文献不存在，或已被删除")
        return item

    def scope(ids):
        selected = list(dict.fromkeys(ids))
        if not selected:
            raise HTTPException(400, "请明确选择至少一篇论文；全选时请传入全部论文 ID。")
        for pid in selected:
            if paper_or_404(pid)["status"] != "ready":
                raise HTTPException(409, "所选论文尚未完成解析，请等待处理完成或重新上传。")
        return selected

    def require_llm():
        if not cfg.llm_configured:
            raise HTTPException(503, "模型尚未配置。请在项目 .env 中设置 LLM_API_KEY 和 LLM_MODEL，然后重启服务；也可以使用本地证据检索。")

    def selected_model(model_id):
        require_llm()
        try:
            return cfg.provider_config() if model_id is None else cfg.provider_config(model_id)
        except ProviderError as exc:
            raise HTTPException(422, exc.public_message) from None

    def public_settings():
        result = cfg.public()
        for provider, state in app.state.provider_status.items():
            result["provider_status"][provider].update(state)
        return result

    def protect_operation(kind: str):
        """Bound long requests and reject concurrent duplicates without auto-retries."""
        def decorate(fn):
            @wraps(fn)
            async def wrapped(*args, **kwargs):
                payload = {key: value.model_dump() if isinstance(value, BaseModel) else value for key, value in kwargs.items()}
                fingerprint = hashlib.sha256(json.dumps([kind, payload], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                with inflight_lock:
                    if fingerprint in inflight:
                        raise HTTPException(409, "相同操作正在处理中，请等待当前请求完成后再重试。", headers={"X-EvidenceLab-Error-Code": "request_in_progress"})
                    inflight.add(fingerprint)
                try:
                    # Plan/evidence stages each have one bounded protocol repair;
                    # verification can additionally align fixed scope fields.
                    # Keep the whole operation below the UI's 400 second limit.
                    stages = 6 if kind == "verify" else 4 if kind == "qa" else 2
                    deadline = 600 if kind == "extract" else min(390, max(30, cfg.timeout * stages + 15))
                    async with asyncio.timeout(deadline):
                        return await fn(*args, **kwargs)
                except TimeoutError:
                    raise HTTPException(504, "处理超过时间限制，本次未保存新结果。请缩小文献范围后重试；已有实验和报告仍保留。", headers={"X-EvidenceLab-Error-Code": "request_timeout"}) from None
                finally:
                    with inflight_lock:
                        inflight.discard(fingerprint)
            return wrapped
        return decorate

    async def provider_call(fn, *args):
        try:
            args = list(args)
            config = args[-1] if args and isinstance(args[-1], dict) and "model" in args[-1] else None
            if config is not None:
                config = {**config, "_model_runs": []}
                args[-1] = config
            result = await fn(*args)
            if config is not None:
                for item in result if isinstance(result, list) else [result]:
                    item["model_id"] = config["model"]
                    item["model_runs"] = config["_model_runs"]
                    if any(r.get("compatibility_retry") for r in config["_model_runs"]):
                        item.setdefault("notes" if isinstance(result, list) else "warnings", []).append(
                            "服务商不支持response_format，已保留相同模型重试一次JSON指令模式；响应仍经过完整结构与来源校验。")
            # Offline/no-evidence answers do not verify service connectivity.
            if result and (isinstance(result, list) or result.get("mode") == "llm" and result.get("citations")):
                app.state.provider_status["chat"] = {"status": "available", "message": "本次运行已有模型请求成功；后续请求仍可能失败。", "checked_at": now()}
            return result
        except HTTPException:
            raise
        except ProviderError as exc:
            logger.warning("Provider request failed (%s, %s)", fn.__name__, exc.code)
            # Grounding failures describe model output, not a transport failure.
            if exc.code in {"authentication", "quota", "http_error", "timeout", "connection", "configuration"}:
                app.state.provider_status["chat"] = {"status": "failed", "message": exc.public_message, "error_code": exc.code, "checked_at": now()}
            raise HTTPException(504 if exc.code == "timeout" else exc.status_code, exc.public_message, headers={"X-EvidenceLab-Error-Code": exc.code}) from None
        except (ImportError, ModuleNotFoundError):
            raise HTTPException(503, "所需研究组件尚未安装。请执行安装研究依赖的命令后重启服务。") from None
        except Exception:
            # Provider exceptions can contain keys or full prompts. Return no raw exception.
            logger.warning("Provider request failed (%s)", fn.__name__)
            raise HTTPException(502, "模型服务调用失败。请检查服务地址、模型名称、密钥、额度及网络连接；本次未生成可核验结果。") from None

    def grounded(result: dict, approved: list[dict]) -> dict:
        known = {e["id"]: e for e in approved}
        valid = []
        rejected = False
        for citation in result.get("citations", []):
            eid = citation if isinstance(citation, str) else citation.get("id")
            if eid in known:
                if eid not in {e["id"] for e in valid}:
                    valid.append(known[eid])
            else:
                rejected = True
        result["citations"] = valid
        result.setdefault("warnings", [])
        if rejected:
            result["warnings"].append("已剔除无法追溯到所选论文的引用，本次结果需要复核。")
            result["verdict"] = "unverified" if "answer" in result else "insufficient"
        if not valid and result.get("verdict") not in ("insufficient", "unverified"):
            result["verdict"] = "unverified" if "answer" in result else "insufficient"
            result["warnings"].append("没有可定位的原文证据，不能据此确认结论。")
        return result

    async def ingest(content: bytes, filename: str, is_demo: bool = False) -> dict:
        from .analysis import extract_experiments
        from .parsing import parse_document
        digest = hashlib.sha256(content).hexdigest()
        existing = store.find_by_hash(digest)
        if existing:
            return existing
        suffix = Path(filename).suffix.lower()
        if suffix not in (".pdf", ".txt", ".md"):
            raise DocumentError("仅支持 PDF、TXT 和 Markdown 文件")
        if not content.strip():
            raise DocumentError("文件为空")
        if suffix == ".pdf" and not content.lstrip().startswith(b"%PDF-"):
            raise DocumentError("文件内容不是有效的 PDF")
        pid = uuid.uuid4().hex
        path = store.files / f"{pid}{suffix}"
        try:
            path.write_bytes(content)
            parsed = await run_in_threadpool(parse_document, path, pid, "pymupdf" if is_demo else cfg.parser)
            evidence = parsed.get("evidence", [])
            if not evidence:
                raise DocumentError("未提取到可检索文字。扫描版 PDF 请先使用外部 OCR 生成可检索 PDF 后重试。")
            experiments = extract_experiments(evidence, pid)
            if is_demo:
                for experiment in experiments:
                    experiment["extraction_method"] = "demo"
            paper = {
                "id": pid, "title": parsed.get("title") or Path(filename).stem,
                "filename": filename, "authors": parsed.get("authors", []), "year": parsed.get("year"),
                "page_count": parsed.get("page_count", 1), "status": "ready", "parser": parsed.get("parser", cfg.parser),
                "created_at": now(), "evidence_count": len(evidence), "experiment_count": len(experiments),
                "is_demo": is_demo, "warnings": parsed.get("warnings", []), "file_url": f"/api/papers/{pid}/file", "error": None,
            }
            if is_demo:
                paper["warnings"] = ["合成演示材料，仅用于演示软件流程，不代表真实科研结果。", *paper["warnings"]]
            experiments = prepare_experiments(experiments, paper, evidence)
            if suffix == ".pdf" and not is_demo:
                experiments = await run_in_threadpool(verify_pdf_measurements, experiments, paper, evidence, path, digest)
                experiments = prioritize_measurements(experiments)
            try:
                store.save_paper(paper, digest, path, evidence, experiments)
            except Exception:
                duplicate = store.find_by_hash(digest)
                if not duplicate:
                    raise
                path.unlink(missing_ok=True)
                return duplicate
            return paper
        except Exception:
            path.unlink(missing_ok=True)
            raise

    @app.get("/api/health")
    def health():
        public = public_settings()
        return {"status": "ok", "version": VERSION, "providers": {
            **{k: public[k] for k in ("llm_configured", "langgraph_available")},
            "llm_status": public["provider_status"]["chat"]["status"],
        }}

    @app.get("/api/settings")
    def settings():
        return public_settings()

    @app.get("/api/papers")
    def papers():
        return store.list_papers()

    @app.post("/api/papers/upload")
    async def upload(files: list[UploadFile] = File(...)):
        if not files or len(files) > 10:
            raise HTTPException(400, "每次请上传 1—10 个文件")
        accepted, errors = [], []
        for file in files:
            filename = (file.filename or "untitled.txt").replace("\\", "/").split("/")[-1]
            try:
                content = await file.read(MAX_FILE_BYTES + 1)
                if len(content) > MAX_FILE_BYTES:
                    raise DocumentError("单个文件不能超过 32 MB")
                accepted.append(await ingest(content, filename))
            except DocumentError as exc:
                errors.append({"filename": filename, "error": str(exc)})
            except ImportError:
                errors.append({"filename": filename, "error": "所选解析组件未安装，请安装研究依赖或使用 PyMuPDF。"})
            except Exception:
                errors.append({"filename": filename, "error": "文档解析失败。请检查文件是否加密、损坏，或配置的解析器是否可用。"})
            finally:
                await file.close()
        return {"papers": accepted, "errors": errors}

    @app.post("/api/demo/seed")
    async def seed_demo():
        result = []
        demo_root = ROOT / "examples"
        for filename in ("study_a.md", "study_b.md", "study_c.md"):
            result.append(await ingest((demo_root / filename).read_bytes(), filename, True))
        return {"papers": result}

    @app.get("/api/papers/{paper_id}")
    def paper_detail(paper_id: str):
        return paper_or_404(paper_id)

    @app.get("/api/papers/{paper_id}/file")
    def paper_file(paper_id: str):
        paper = paper_or_404(paper_id)
        try:
            path = store.get_file(paper_id)
        except ValueError:
            raise HTTPException(409, "原始文件路径未通过目录安全检查，请使用路径迁移工具检查；已有研究数据未修改。") from None
        if path is None or not path.is_file():
            raise HTTPException(404, "原始文件不存在")
        return FileResponse(path, media_type="application/pdf" if path.suffix == ".pdf" else "text/plain; charset=utf-8", content_disposition_type="inline", filename=paper["filename"], headers={"X-Content-Type-Options": "nosniff"})

    @app.get("/api/papers/{paper_id}/pages/{page_number}/preview")
    async def paper_page_preview(paper_id: str, page_number: int):
        paper = paper_or_404(paper_id)
        if not 1 <= page_number <= paper["page_count"]:
            raise HTTPException(422, "预览页码超出文献范围。")
        try:
            path = store.get_file(paper_id)
        except ValueError:
            raise HTTPException(409, "原始文件路径未通过目录安全检查，无法生成页面预览。") from None
        if path is None or not path.is_file():
            raise HTTPException(404, "原始文件不存在，保存的摘录仍可阅读。")
        if path.suffix.lower() != ".pdf":
            raise HTTPException(415, "页面图片预览仅支持 PDF，请使用原始文件入口。")

        def generate():
            from .pdf_preview import render_pdf_page
            with path.open("rb") as source:
                content = source.read(MAX_FILE_BYTES + 1)
            if len(content) > MAX_FILE_BYTES:
                raise HTTPException(413, "原文超过页面预览大小上限。")
            if hashlib.sha256(content).hexdigest() != store.get_file_hash(paper_id):
                raise HTTPException(409, "原始文件 SHA256 与入库记录不一致，已停止预览；已有研究数据未修改。")
            return render_pdf_page(content, page_number)

        try:
            async with preview_slots:
                content = await run_in_threadpool(generate)
        except DocumentError as exc:
            raise HTTPException(422, str(exc)) from None
        except FileNotFoundError:
            raise HTTPException(404, "原始文件不存在，保存的摘录仍可阅读。") from None
        except ImportError:
            raise HTTPException(503, "当前缺少 PDF 页面渲染组件，请使用原始文件入口。") from None
        except OSError:
            raise HTTPException(422, "无法读取原始文件，请使用原始文件入口检查文献。") from None
        return Response(content, media_type="image/png", headers={"Cache-Control": "private, max-age=60",
                                                                   "X-Content-Type-Options": "nosniff"})

    @app.delete("/api/papers/{paper_id}")
    def delete_paper(paper_id: str):
        paper_or_404(paper_id)
        path = store.get_file(paper_id)
        store.delete_paper(paper_id)
        if path:
            path.unlink(missing_ok=True)
        return {"deleted": True}

    @app.patch("/api/experiments/{experiment_id}")
    def edit_experiment(experiment_id: str, patch: ExperimentPatch):
        experiment = store.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(404, "实验记录不存在")
        previous = copy.deepcopy(experiment)
        updates = patch.model_dump(exclude_unset=True)
        expected_revision = updates.pop("expected_revision", None)
        if expected_revision is not None and store.experiment_payload_revision(previous) != expected_revision:
            raise HTTPException(409, "实验记录已由其他页面更新。请保留当前输入并重新载入最新记录后再修改。")
        if not updates:
            return experiment
        if "name" in updates and not updates["name"]:
            raise HTTPException(400, "实验名称不能为空")
        if "conditions" in updates and updates["conditions"] is None:
            updates["conditions"] = {}
        if "evidence_ids" in updates and updates["evidence_ids"] is None:
            updates["evidence_ids"] = []
        if "field_evidence" in updates and updates["field_evidence"] is None:
            updates["field_evidence"] = {}
        valid = {e["id"] for e in store.evidence_for([experiment["paper_id"]])}
        requested = set(updates.get("evidence_ids", []))
        for field_ids in (updates.get("field_evidence") or {}).values():
            requested.update(field_ids)
        if not requested.issubset(valid):
            raise HTTPException(400, "引用必须来自该实验所属论文")
        edited = {field for field, value in updates.items()
                  if field not in {"evidence_ids", "field_evidence"} and value != experiment.get(field)}
        experiment["field_evidence"] = dict(experiment.get("field_evidence", {}))
        experiment["evidence_quotes"] = dict(experiment.get("evidence_quotes", {}))
        for field in edited:
            # Existing model quotes cannot certify a newly typed manual value.
            experiment["evidence_quotes"].pop(field, None)
            if "field_evidence" not in updates:
                experiment["field_evidence"].pop(field, None)
        if any(field in updates and updates[field] != experiment.get(field) for field in ("field_evidence", "evidence_ids")):
            experiment["evidence_quotes"] = {}
        experiment.update(updates)
        experiment.pop("machine_verification", None)  # A manual edit is not a new machine audit.
        experiment.pop("task_verification", None)
        experiment.pop("task_label_zh", None)
        approved_ids = set(experiment.get("evidence_ids", []))
        if any(not set(ids).issubset(approved_ids) for ids in experiment.get("field_evidence", {}).values()):
            raise HTTPException(400, "字段证据必须同时列在该实验的原文证据中")
        if "field_evidence" in updates:
            # User-selected field sources are an explicit association. Attach
            # fresh quotations only when the typed value occurs literally in
            # them; old model quotes never certify a newly edited value.
            from .providers import _field_is_literal, _metadata_measurement_conflicts
            known_sources = {e["id"]: e for e in store.evidence_for([experiment["paper_id"]])}
            for field, identifiers in experiment["field_evidence"].items():
                if experiment["evidence_quotes"].get(field) or not identifiers:
                    continue
                quotations = [{"id": eid, "quote": known_sources[eid]["text"]} for eid in identifiers]
                if (_field_is_literal(field, experiment.get(field), [], quotations)
                        and not _metadata_measurement_conflicts(experiment, field)):
                    experiment["evidence_quotes"][field] = quotations
        experiment["review_required"] = True
        experiment["source_pages"] = sorted({e["page"] for e in store.evidence_for([experiment["paper_id"]])
                                              if e["id"] in approved_ids and type(e.get("page")) is int and e["page"] > 0})
        notes = experiment.setdefault("notes", [])
        notes.append(f"人工修正 {', '.join(updates.keys())}（{now()}）；请结合原文复核。")
        if not store.update_experiment(experiment, expected=previous, expected_revision=expected_revision):
            raise HTTPException(409, "实验记录刚刚被更新或删除，请刷新后重新修改。")
        return experiment

    @app.get("/api/experiments/{experiment_id}/history")
    def experiment_history(experiment_id: str):
        if not store.get_experiment(experiment_id):
            raise HTTPException(404, "实验记录不存在")
        return store.experiment_history(experiment_id)

    @app.post("/api/papers/{paper_id}/manual-candidates")
    def manual_candidate(paper_id: str, body: ManualCandidate):
        """Human entry follows the same grounding and candidate review boundary."""
        from .providers import _checked_extraction
        paper = paper_or_404(paper_id)
        fields = {"task", "method", "dataset", "split", "metric", "value", "unit", "sample_size", "conditions"}
        if (not body.name.strip() or not set(body.fields).issubset(fields)
                or not set(body.evidence_quotes).issubset(fields)
                or body.fields.get("value") is None):
            raise HTTPException(422, "请填写实验名称、有限结果数值和支持它的原文；不确定的字段请留空。")
        raw = {"name": body.name.strip(), **body.fields,
               "field_evidence": {key: [ref.model_dump() for ref in refs]
                                  for key, refs in body.evidence_quotes.items()}}
        revision = store.experiment_revision(paper["experiments"])
        try:
            records = _checked_extraction({"experiments": [raw]}, paper["evidence"], paper_id, {}, 0)
        except ProviderError as exc:
            raise HTTPException(422, exc.public_message.replace("模型输出", "人工候选").replace("模型", "候选")) from None
        if len(records) != 1 or records[0].get("value") is None:
            raise HTTPException(422, "候选缺少有原文依据的结果数值；尚未保存。")
        records[0].update(extraction_method="manual", notes=[
            "人工从原文建立候选，未调用模型；已校验引用、字面值及可检测的测量归属冲突，仍需人工核查完整实验条件。"])
        records = prepare_experiments(records, paper, paper["evidence"])
        candidate = store.save_extraction_candidate(paper_id, records, "manual", revision)
        result = store.get_paper(paper_id)
        result["latest_extraction_candidate_id"] = candidate["id"]
        return result

    @app.post("/api/papers/{paper_id}/extract")
    @protect_operation("extract")
    async def extract(paper_id: str, body: ExtractionRequest):
        from .analysis import extract_experiments
        from .providers import extract_llm
        paper = paper_or_404(paper_id)
        model_config = selected_model(body.model_id) if body.mode == "llm" else None
        sources = paper["evidence"]
        if body.pages and max(body.pages) > paper["page_count"]:
            raise HTTPException(422, "提取页码超出当前文献页数；已有研究记录未修改。")
        parsed = None
        if body.reparse:
            from .parsing import parse_document
            try:
                path = store.get_file(paper_id)
                if path is None or not path.is_file():
                    raise HTTPException(404, "原始文件不存在，无法重新解析；已有证据保持不变。")
                digest = await run_in_threadpool(lambda: hashlib.sha256(path.read_bytes()).hexdigest())
                if digest != store.get_file_hash(paper_id):
                    raise HTTPException(409, "原始文件 SHA256 与入库记录不一致，已停止重新解析；已有研究数据未修改。")
                parsed = await run_in_threadpool(parse_document, path, paper_id, cfg.parser)
            except DocumentError as exc:
                raise HTTPException(422, str(exc)) from None
            except ValueError:
                raise HTTPException(409, "原文路径未通过目录安全检查，无法重新解析。") from None
            except ImportError:
                raise HTTPException(503, "所选解析组件未安装，请选择已安装的解析器。") from None
            sources = parsed.get("evidence", [])
            if not sources:
                raise HTTPException(422, "重新解析没有提取到原文证据，已有资料未修改。")
            original_sources = {e["id"]: e for e in paper["evidence"]}
            # Preserve the old evidence record, while giving the same upgraded
            # parser record a stable ID across repeated reparses. Include its
            # original ID so identical passages at separate occurrences stay
            # separate rather than merging experiments by matching text alone.
            remapped = []
            for source in sources:
                if source["id"] in original_sources and source != original_sources[source["id"]]:
                    canonical = json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    source = {**source, "id": "ev_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]}
                if source["id"] in original_sources and source != original_sources[source["id"]]:
                    raise HTTPException(409, "候选原文标识与已保存证据冲突，请检查解析结果；已有证据未修改。")
                remapped.append(source)
            sources = remapped
        if body.pages is not None:
            sources = [source for source in sources if source.get("page") in body.pages]
            if not sources:
                raise HTTPException(422, "所选页没有可读取原文，请核对页码或重新解析；已有记录未修改。")
        if body.mode == "llm":
            require_llm()
            experiments = await provider_call(extract_llm, sources, paper_id,
                                              {**model_config, "max_results": body.max_results})
        else:
            experiments = await run_in_threadpool(extract_experiments, sources, paper_id)
        if not experiments:
            raise HTTPException(422, "未提取到有效实验，已有记录未修改。")
        try:
            experiments = prepare_experiments(experiments, paper, sources)
        except ProviderError as exc:
            raise HTTPException(exc.status_code, exc.public_message) from None
        if body.mode == "rule":
            experiments = await run_in_threadpool(verify_pdf_measurements, experiments, paper, sources,
                                                   store.get_file(paper_id), store.get_file_hash(paper_id))
            experiments = prioritize_measurements(experiments)
        truncated = len(experiments) > body.max_results
        experiments = experiments[:body.max_results]
        try:
            candidate = store.save_extraction_candidate(paper_id, experiments, body.mode, store.experiment_revision(paper["experiments"]),
                                                        source_evidence=sources if parsed is not None else None,
                                                        parser=parsed.get("parser") if parsed else None,
                                                        extraction_scope={"pages": body.pages, "total_pages": paper["page_count"],
                                                                          "max_results": body.max_results,
                                                                          "truncated": truncated,
                                                                          "completeness": "not_assessed"})
        except Exception:
            if not store.get_paper(paper_id):
                raise HTTPException(409, "提取期间文献已删除，本次候选未保存。") from None
            raise
        result = paper_or_404(paper_id)
        result["latest_extraction_candidate_id"] = candidate["id"]
        return result

    @app.get("/api/papers/{paper_id}/extraction-candidates")
    def extraction_candidates(paper_id: str):
        return paper_or_404(paper_id)["extraction_candidates"]

    @app.post("/api/papers/{paper_id}/extraction-candidates/{candidate_id}/accept")
    def accept_candidate(paper_id: str, candidate_id: str, body: CandidateAcceptance):
        paper_or_404(paper_id)
        try:
            store.accept_extraction_candidate(paper_id, candidate_id, body.experiment_ids)
        except KeyError:
            raise HTTPException(404, "候选结果不存在") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        return paper_or_404(paper_id)

    @app.post("/api/papers/{paper_id}/extraction-candidates/{candidate_id}/reject")
    def reject_candidate(paper_id: str, candidate_id: str):
        paper_or_404(paper_id)
        try:
            store.reject_extraction_candidate(paper_id, candidate_id)
        except KeyError:
            raise HTTPException(404, "候选结果不存在") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        return paper_or_404(paper_id)

    @app.post("/api/qa")
    @protect_operation("qa")
    async def qa(body: QARequest):
        from .analysis import answer_local
        from .providers import answer_llm, verify_claim
        ids = scope(body.paper_ids)
        evidence = store.evidence_for(ids)
        if body.engine == "local":
            return grounded(await run_in_threadpool(answer_local, body.question, evidence), evidence)
        require_llm()
        if body.engine == "llm":
            task_config = selected_model(body.model_id)
            result = grounded(await provider_call(answer_llm, body.question, evidence, task_config), evidence)
            if result.get("citations") and result.get("verdict") != "insufficient":
                try:
                    assertions = result.get("assertion_evidence", [])
                    claim_text = "\n\n".join(item["text"] for item in assertions
                                              if isinstance(item, dict) and isinstance(item.get("text"), str)
                                              and item["text"].strip()) if isinstance(assertions, list) else ""
                    # Re-retrieve from the selected library scope. Restricting
                    # this to the answer's chosen rows discards table headers,
                    # captions and counter-evidence and repeats the same error.
                    check = grounded(await provider_call(verify_claim, claim_text or result["answer"], evidence,
                                                         task_config), evidence)
                except HTTPException as exc:
                    # Retain the already grounded answer when only the separate
                    # semantic check fails; its uncertainty must remain visible.
                    result["verdict"] = "unverified"
                    result["verification"] = {"verdict": "insufficient", "status": "failed", "explanation": exc.detail}
                    result["warnings"].append(f"回答已完成原文引用检查，但独立核验未完成：{exc.detail} 请人工复核。")
                    return result
                result["verification"] = check
                result["traces"] = [*result.get("traces", []), *check.get("traces", [])]
                if check.get("verdict") not in ("supported", "partial"):
                    result["verdict"] = "unverified"
                    result["warnings"].append("独立核验未确认回答得到原文支持，请查看引用并人工复核。")
                elif check.get("verdict") == "supported" and not result.get("withheld_assertions"):
                    result["verdict"] = "evidence_found"
                else:
                    result["verdict"] = "unverified"
                    result["warnings"].append("独立核验仅确认部分支持，请查看限定条件。")
            return result

    @app.post("/api/verify")
    @protect_operation("verify")
    async def verify(body: VerifyRequest):
        from .providers import verify_claim
        ids = scope(body.paper_ids)
        evidence = store.evidence_for(ids)
        if body.engine == "llm":
            require_llm()
        task_config = selected_model(body.model_id) if body.engine == "llm" else None
        result = grounded(await provider_call(verify_claim, body.claim, evidence, task_config), evidence)
        return result

    @app.post("/api/compare")
    @protect_operation("compare")
    async def compare(body: ScopeRequest):
        from .orchestration import run_comparison
        ids = scope(body.paper_ids)
        if len(ids) < 2:
            raise HTTPException(400, "请选择至少两篇论文进行比较")
        experiments = store.experiments_for(ids)
        if len(experiments) > 150:
            raise HTTPException(400, "单次比较最多支持 150 项实验，请缩小文献范围")
        per_paper = {paper_id: sum(e["paper_id"] == paper_id for e in experiments) for paper_id in ids}
        pairs = (len(experiments) ** 2 - sum(count ** 2 for count in per_paper.values())) // 2
        if pairs > 2000:
            raise HTTPException(422, "所选实验超过单次 2000 组跨论文比较上限，请减少参与论文后重试；本次未保存报告。")
        evidence = store.evidence_for(ids)
        try:
            graph = await run_in_threadpool(run_comparison, experiments, evidence)
        except (ImportError, ModuleNotFoundError):
            raise HTTPException(503, "当前缺少比较流程组件，无法生成报告；请检查部署依赖。已有研究数据未修改。") from None
        except Exception:
            logger.warning("Comparison workflow failed")
            raise HTTPException(500, "比较流程执行失败，本次未保存新报告；已有实验和历史报告仍保留，请重试或检查服务日志。") from None
        paper_list = [{k: v for k, v in paper_or_404(pid).items() if k not in ("evidence", "experiments", "extraction_candidates", "experiment_revisions")} for pid in ids]
        report = {"id": uuid.uuid4().hex, "papers": paper_list, "experiments": experiments,
                  "comparisons": graph["comparisons"], "traces": graph["traces"], "created_at": now(),
                  "mode": "rules", "evidence": evidence, "orchestrator": "langgraph",
                  "warnings": ["条件对齐与差异分析为规则流程；潜在冲突仍需原文复核，不代表科学真伪判断。"]}
        if not store.save_report(report):
            raise HTTPException(409, "分析期间文献或实验已更新，本次旧结果未保存；请重新生成报告。")
        return report

    @app.get("/api/reports")
    def reports():
        return store.list_reports()

    @app.get("/api/reports/{report_id}")
    def report_detail(report_id: str):
        report = store.get_report(report_id, include_stale=True)
        if not report:
            raise HTTPException(404, "报告不存在")
        return report

    @app.get("/api/reports/{report_id}/export")
    def export_report(report_id: str, format: Literal["markdown", "json"] = "markdown", include_stale: bool = False):
        report = store.get_report(report_id, include_stale=include_stale)
        if not report:
            raise HTTPException(404, "报告不存在或相关文献已更新，请重新运行分析")
        headers = {"Content-Disposition": f'attachment; filename="evidencelab-{report_id[:8]}.{ "json" if format == "json" else "md"}"'}
        if format == "json":
            return Response(json.dumps(report, ensure_ascii=False, indent=2), media_type="application/json", headers=headers)
        def safe(value):
            return str(value if value is not None else "未报告").replace("|", "\\|").replace("\n", " ")
        paper_names = {p["id"]: p["title"] for p in report["papers"]}
        lines = ["# 文献证据对比报告", "", f"生成时间：{report['created_at']}", "", "分析方式：LangGraph 编排的规则流程，未进行科学真伪或统计显著性判断。", ""]
        if report.get("stale"):
            lines.extend(["**历史快照：" + report["stale_reason"] + "**", ""])
        if any(p["is_demo"] for p in report["papers"]):
            lines.extend(["**包含合成演示材料，不能作为科研证据。**", ""])
        lines.extend(["## 实验矩阵", "", "|论文 / 实验|任务|方法|数据集|划分|指标|数值|单位|样本量|条件|证据|", "|---|---|---|---|---|---|---|---|---|---|---|"])
        for exp in report["experiments"]:
            conditions = "; ".join(f"{k}={v}" for k, v in exp.get("conditions", {}).items()) or "未报告"
            lines.append("|" + "|".join(safe(v) for v in [paper_names[exp["paper_id"]] + " / " + exp["name"], ((exp.get("task_label_zh") + " / " + exp["task"]) if exp.get("task_label_zh") and exp.get("task") else exp.get("task") or "任务尚未自动确认"), exp.get("method"), exp.get("dataset"), exp.get("split"), exp.get("metric"), exp.get("value"), exp.get("unit"), exp.get("sample_size"), conditions, ", ".join(exp.get("evidence_ids", []))]) + "|")
        lines.extend(["", "## 字段原文与复核", "", "以下引文用于定位字段来源；字面出现不能代替实验归属和语义复核。任务尚未自动确认表示尚未建立逐实验来源归属，不表示论文没有报告任务。其他未确认字段没有被补齐。", ""])
        for exp in report["experiments"]:
            review_label = ("测量字段机器核验通过；协议未核验；未进行人工审核" if machine_measurement_verified(exp)
                            else "需人工复核" if exp.get("review_required", True) else "请确认复核记录")
            lines.extend([f"### {safe(exp['name'])}", "", f"提取方式：{safe(exp.get('extraction_method'))}；复核状态：{review_label}。", ""])
            if machine_measurement_verified(exp):
                check = exp["machine_verification"]
                lines.extend(["测量字段机器核验通过（未进行人工审核）：方法、数据集、指标、主值及单位与原始 PDF 行列/表注相符。实验协议与统计结论不在认证范围。",
                              f"完整结果单元格：{safe(check['raw_cell'])}；PDF SHA256：{check['pdf_sha256']}。", ""])
            if machine_task_verified(exp, report["evidence"]):
                lines.extend(["任务归属机器核验通过（独立于测量范围）：" + safe(exp.get("task_label_zh") or exp["task"]) + "；不认证完整实验协议或跨论文可比性。", ""])
            for field, references in exp.get("evidence_quotes", {}).items():
                for ref in references:
                    lines.extend([f"- {safe(field)} · {safe(ref.get('id'))}", "", "> " + str(ref.get("quote", "")).replace("\n", "\n> "), ""])
            for note in exp.get("notes", []):
                lines.append(f"- 复核说明：{safe(note)}")
        lines.extend(["", "## 条件与差异", ""])
        for comp in report["comparisons"]:
            lines.extend([f"- {comp['left_id']} ↔ {comp['right_id']}：{comp['comparability']} / {comp['verdict']}", *[f"  - {safe(r)}" for r in [*comp.get("reasons", []), *comp.get("differences", [])]]])
        lines.extend(["", "## 原文证据", ""])
        referenced = {eid for exp in report["experiments"] for eid in exp.get("evidence_ids", [])}
        referenced.update(eid for comp in report["comparisons"] for eid in comp.get("evidence_ids", []))
        for ev in report["evidence"]:
            if ev["id"] not in referenced:
                continue
            link = f"/api/papers/{ev['paper_id']}/file#page={ev['page']}"
            lines.extend([f"### {ev['id']}", "", f"{paper_names[ev['paper_id']]} · 第 {ev['page']} 页 · [原文]({link})", "", "> " + ev["text"].replace("\n", "\n> "), ""])
        return Response("\n".join(lines), media_type="text/markdown; charset=utf-8", headers=headers)

    @app.get("/api/{rest:path}")
    def api_missing(rest: str):
        raise HTTPException(404, "接口不存在")

    dist = ROOT / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    else:
        @app.get("/")
        def unbuilt():
            return JSONResponse({"message": "后端已启动。请构建前端，或运行 npm run dev 后访问 http://localhost:5173", "api_docs": "/docs"})
    return app


app = create_app()
