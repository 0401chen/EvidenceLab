"""Opt-in online preview: independent library, private local config, no probes.

Unlike serve_offline.py this enables ordinary model networking, but nothing
calls a model at startup. The listening socket is reserved before any DB writes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
OWNER = "EvidenceLab-online-preview-v1"


class PreviewError(ValueError):
    """Fixed local message that is safe to print without provider/config data."""



def preview_directory(path: Path) -> Path:
    base = ROOT / "artifacts" / "previews"
    data = path.absolute()
    if not data.is_relative_to(base) or data == base:
        raise PreviewError("Use a separate library below artifacts/previews/.")
    # Reject junctions too: a local alias must not lead into the demo library.
    for component in [base, *data.parents, data]:
        if component == ROOT or ROOT not in component.parents:
            continue
        if (component.is_symlink() or getattr(component, "is_junction", lambda: False)()
                or component.exists() and component.resolve() != component.absolute()):
            raise PreviewError("Preview paths must not contain symlinks or junctions.")
    data = data.resolve()
    if not data.is_relative_to(base.resolve()) or data == base.resolve():
        raise PreviewError("Preview directory escapes its isolated boundary.")
    if data.exists() and any(data.iterdir()):
        marker = data / ".preview-owner.json"
        try:
            info = json.loads(marker.read_text(encoding="utf8"))
        except (OSError, ValueError):
            raise PreviewError("Existing files lack the online-preview ownership marker; choose a new folder.") from None
        if info != {"owner": OWNER, "data_dir": str(data)}:
            raise PreviewError("Preview ownership marker does not match this library.")
    return data


def prepare_config(data: Path, config_file: Path | None, allowed_models: str | None = None):
    from dotenv import dotenv_values

    if config_file is not None:
        if not config_file.is_file():
            raise PreviewError("Private config file is missing; supply --config-file or use process environment.")
        values = dotenv_values(config_file, encoding="utf-8-sig", interpolate=False)
        # An explicitly selected file owns provider credentials and endpoint.
        # Never import its data directory or mix it with stale inherited keys.
        os.environ.update(
            LLM_API_KEY=values.get("LLM_API_KEY") or values.get("OPENAI_API_KEY") or "",
            OPENAI_API_KEY="",
            LLM_BASE_URL=values.get("LLM_BASE_URL", values.get("OPENAI_BASE_URL", "https://api.openai.com/v1")) or "",
            OPENAI_BASE_URL="",
            LLM_MODEL=values.get("LLM_MODEL") or "",
            LLM_TIMEOUT=values.get("LLM_TIMEOUT") or "90",
            LLM_ALLOWED_MODELS=values.get("LLM_ALLOWED_MODELS") or "",
        )
    if allowed_models is not None:
        os.environ["LLM_ALLOWED_MODELS"] = allowed_models
    # Must precede importing main: its module-level app must also use this DB.
    os.environ.update(EVIDENCELAB_DATA_DIR=str(data), EVIDENCE_PARSER="pymupdf",
                      PYTHON_DOTENV_DISABLED="1", PYTHONDONTWRITEBYTECODE="1")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from backend.app.config import Config
    from backend.app.model_catalog import valid_model_id
    cfg = Config.from_env(data)
    if any(not valid_model_id(m) for m in cfg.allowed_models):
        raise PreviewError("Invalid server-enabled model ID; no model request was sent.")
    return cfg


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/previews/online")
    parser.add_argument("--config-file", type=Path)
    parser.add_argument("--allowed-models", help="Explicit comma-separated IDs actually authorized by this provider.")
    parser.add_argument("--frontend-dir", type=Path, default=ROOT / "artifacts/preview-build/frontend-dist")
    parser.add_argument("--check", action="store_true", help="Read-only configuration/port check; no DB or model request.")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535 or args.port in {8000, 8790, 8792, 8793}:
        parser.error("Use a separate preview port; 8000/8790/8792/8793 are protected.")
    try:
        data = preview_directory(args.data_dir)
        frontend = args.frontend_dir.resolve()
        if not frontend.is_relative_to((ROOT / "artifacts").resolve()) or not (frontend / "index.html").is_file():
            raise PreviewError("Build the preview frontend under artifacts first (START_PREVIEW_AI.ps1 does this).")
        config_file = args.config_file
        if config_file is None and (ROOT / ".env").is_file():
            config_file = ROOT / ".env"
        with socket.socket() as listener:
            if os.name == "nt":
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                listener.bind(("127.0.0.1", args.port))
            except OSError:
                raise PreviewError("Preview port is occupied/unavailable; choose another port. No process was stopped.") from None
            cfg = prepare_config(data, config_file, args.allowed_models)
            print(json.dumps({"preview": "online", "url": f"http://127.0.0.1:{args.port}",
                              "data_dir": str(data), "model": cfg.model,
                              "configured": cfg.llm_configured, "provider": cfg.public()["llm_base_url"],
                              "connection": "untested" if cfg.llm_configured else "unconfigured",
                              "startup_model_requests": 0}, ensure_ascii=True), flush=True)
            if args.check:
                return
            data.mkdir(parents=True, exist_ok=True)
            marker = data / ".preview-owner.json"
            if not marker.exists():
                marker.write_text(json.dumps({"owner": OWNER, "data_dir": str(data)}), encoding="utf8")
            from backend.app.main import app
            from fastapi.staticfiles import StaticFiles
            app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) not in {"", "/"}]
            app.mount("/", StaticFiles(directory=frontend, html=True), name="online-preview-frontend")
            import uvicorn
            listener.listen(128)
            uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port,
                                         log_level="warning", access_log=False)).run(sockets=[listener])
    except PreviewError as exc:
        parser.exit(1, f"Online preview refused: {exc}\n")
    except (OSError, ValueError):
        # Never print library exceptions: they can contain config bytes or URLs.
        parser.exit(1, "Online preview refused: check local paths and permissions; no model probe was sent.\n")


if __name__ == "__main__":
    main()
