from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

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
            model=os.getenv("LLM_MODEL", ""),
            parser=os.getenv("EVIDENCE_PARSER", "pymupdf").lower(),
            timeout=max(5.0, min(timeout, 180.0)),
        )

    @property
    def llm_configured(self) -> bool:
        return bool(self.api_key.strip() and self.model.strip())

    def provider_config(self) -> dict:
        return {"api_key": self.api_key, "base_url": self.base_url,
                "model": self.model, "timeout": self.timeout}

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
        return {
            "llm_configured": self.llm_configured,
            "llm_model": self.model,
            "llm_base_url": self._public_endpoint(self.base_url),
            "parser": self.parser,
            "qa_engine": "local",
            "provider_status": {
                "chat": {"configured": self.llm_configured, "status": "untested" if self.llm_configured else "unconfigured",
                         "message": "已配置，尚未验证实际请求。" if self.llm_configured else "尚未配置模型。"},
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
