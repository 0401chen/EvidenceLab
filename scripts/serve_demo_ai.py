"""Serve an existing rehearsal copy with ordinary model networking enabled."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare(data: Path, config_file: Path | None = None) -> list[str]:
    from dotenv import dotenv_values

    data = data.resolve()
    database = data / "evidencelab.sqlite3"
    if not database.is_file():
        raise ValueError("Existing demo database required; restore a verified backup into a new folder first.")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Demo database integrity check failed.")
        if not connection.execute("SELECT count(*) FROM papers").fetchone()[0]:
            raise ValueError("Demo library is empty; restore the reviewed library copy first.")
    # An explicitly selected local file is authoritative. Never copy it or echo
    # its contents, and never take a data-directory override from that file.
    if config_file:
        if not config_file.is_file():
            raise ValueError("Config file does not exist; supply -ConfigFile or use environment variables.")
        values = dotenv_values(config_file)
        for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_TIMEOUT"):
            os.environ[name] = values.get(name) or ""
    os.environ.update(EVIDENCELAB_DATA_DIR=str(data), EVIDENCE_PARSER="pymupdf")
    return [name for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
            if not os.environ.get(name, "").strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config-file", type=Path)
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument("--audit-dir", type=Path,
                        help="Optional private local acceptance capture under artifacts/; not required for AI.")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be between 1024 and 65535.")
    try:
        missing = prepare(args.data_dir, args.config_file)
    except (ValueError, sqlite3.Error):
        parser.exit(1, "Cannot start: verify existing demo database and config-file path; source files unchanged.\n")
    if missing:
        print("Model configuration incomplete: " + ", ".join(missing)
              + ". Fill these in your local config; local evidence mode remains available.", flush=True)
        # Incomplete configuration must never accidentally activate a partial
        # key/model inherited from another process or a default provider URL.
        os.environ.update(LLM_API_KEY="", OPENAI_API_KEY="", LLM_MODEL="")
    sys.path.insert(0, str(ROOT))
    from backend.app.main import app
    if args.audit_dir:
        from demo_model_audit import install
        install(app, args.audit_dir)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
