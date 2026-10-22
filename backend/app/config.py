from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from .model_catalog import DEFAULT_MODEL, valid_model_id

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)


@dataclass(frozen=True)
class Config:
    data_dir: Path
    api_key: str
    base_url: str
    model: str
    parser: str
    timeout: float = 90.0
    allowed_models: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, data_dir: str | Path | None = None) -> "Config":
        try:
            timeout = float(os.getenv("LLM_TIMEOUT", "90"))
            if not math.isfinite(timeout):
                timeout = 90.0
        except (ValueError, TypeError):
            timeout = 90.0
        return cls(
            data_dir=Path(data_dir or os.getenv("EVIDENCELAB_DATA_DIR", ROOT / "data")).resolve(),
            api_key=os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            base_url=os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/"),
            model=os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODEL,
            parser=os.getenv("EVIDENCE_PARSER", "pymupdf").lower(),
            timeout=max(5.0, min(timeout, 180.0)),
            allowed_models=tuple(dict.fromkeys(m.strip() for m in os.getenv("LLM_ALLOWED_MODELS", "").split(",") if m.strip())),
        )

    @property
    def llm_configured(self) -> bool:
        try:
            endpoint = urlsplit(self.base_url)
            return bool(self.api_key.strip() and valid_model_id(self.model)
                        and endpoint.scheme in {"http", "https"} and endpoint.hostname
                        and not (endpoint.username or endpoint.password or endpoint.query or endpoint.fragment)
                        and (endpoint.port is None or 1 <= endpoint.port <= 65535))
        except ValueError:
            return False

    def provider_config(self, model_id: str | None = None) -> dict:
        from .model_catalog import valid_model_id
        from .errors import ProviderError
        model = model_id if model_id is not None else self.model
        if not valid_model_id(model) or model not in (self.model, *self.allowed_models):
            raise ProviderError("configuration", "所选模型未获服务器启用，请检查型号和服务商权限；不会切回默认模型。")
        return {"api_key": self.api_key, "base_url": self.base_url,
                "model": model, "timeout": self.timeout}

    @staticmethod
    def _public_endpoint(value: str) -> str:
        try:
            endpoint = urlsplit(value)
            hostname = endpoint.hostname or ""
            if ":" in hostname:
                hostname = f"[{hostname}]"
            port = f":{endpoint.port}" if endpoint.port else ""
            # Paths, user-info and query strings can contain provider credentials.
            return f"{endpoint.scheme}://{hostname}{port}" if hostname else "地址格式不正确"
        except ValueError:
            return "地址格式不正确"

    def public(self) -> dict:
        from .model_catalog import CATALOG, valid_model_id
        enabled = {m for m in (self.model, *self.allowed_models) if valid_model_id(m)}
        return {
            "llm_configured": self.llm_configured,
            "llm_model": self.model,
            "llm_base_url": self._public_endpoint(self.base_url),
            "models": [{"id": m, "enabled": bool(self.llm_configured and m in enabled),
                        "default": m == self.model, "availability": "untested",
                        "price_notice": "请查看服务商报价；配置启用不代表已验证访问权限。"}
                       for m in dict.fromkeys([m for m in (self.model, *self.allowed_models, *CATALOG) if valid_model_id(m)])],
            "model_price_notice": "费用由当前实际API服务商计收，请查看服务商报价；本机不自动探测模型。",
            "parser": self.parser,
            "qa_engine": "llm" if self.llm_configured else "local",
            "provider_status": {
                "chat": {"configured": self.llm_configured, "status": "untested" if self.llm_configured else "unconfigured",
                         "message": "已配置，尚未验证实际请求。" if self.llm_configured else "尚未连接模型服务。默认型号不代表已连接。"},
            },
            "langgraph_available": importlib.util.find_spec("langgraph") is not None,
            "capabilities": {
                "local_retrieval": True,
                "pdf_parser": "PyMuPDF",
                "llm_qa": self.llm_configured,
                "llm_verification": self.llm_configured,
                "offline_comparison": True,
            },
        }
