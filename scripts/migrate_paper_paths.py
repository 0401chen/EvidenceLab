"""Verify and migrate paper originals after moving an EvidenceLab data directory.

The default is a dry run. Each run first creates and checks a SQLite backup;
only an explicitly requested, fully verified transaction changes file_path.
Reports deliberately contain no paper identifiers, names, contents or digests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath


DATABASE_NAME = "evidencelab.sqlite3"


class _ValidationError(Exception):
    """An anonymous, safe-to-report validation error code."""


def _connection(path: Path, *, writable: bool = False) -> sqlite3.Connection:
    # URI mode prevents sqlite3.connect from silently creating a missing source.
    mode = "rw" if writable else "ro"
    db = sqlite3.connect(f"{path.as_uri()}?mode={mode}", uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def _backup(database: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = directory / f"evidencelab.before-paper-path-migration.{stamp}.{uuid.uuid4().hex}.sqlite3"
    # Reserve a new file exclusively. Never overwrite an earlier backup.
    descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    source = _connection(database)
    target = sqlite3.connect(destination)
    try:
        source.backup(target)
        if target.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise _ValidationError("backup_integrity_failed")
    finally:
        target.close()
        source.close()
    return destination


def _read_rows(db: sqlite3.Connection) -> list[tuple[str, str, str]]:
    required = {"papers", "evidence", "experiments", "reports"}
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not required.issubset(tables):
        raise _ValidationError("unsupported_database_schema")
    columns = {row[1] for row in db.execute("PRAGMA table_info(papers)")}
    if not {"id", "sha256", "payload", "file_path"}.issubset(columns):
        raise _ValidationError("unsupported_database_schema")
    return [tuple(row) for row in db.execute("SELECT id,sha256,file_path FROM papers ORDER BY rowid")]


def _check_spelling(raw: str) -> None:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise _ValidationError("invalid_path")
    # Reject ambiguous Windows spellings even if their normalized path looks safe.
    if "/" in raw and "\\" in raw:
        raise _ValidationError("mixed_path_separators")
    if any(part in (".", "..") for part in re.split(r"[\\/]", raw)):
        raise _ValidationError("path_traversal")
    windows = PureWindowsPath(raw)
    if windows.drive.startswith("\\\\") or raw.startswith("//"):
        raise _ValidationError("unc_path_not_supported")
    if windows.drive and not windows.is_absolute():
        raise _ValidationError("drive_relative_path")
    # An alternate data stream is not a paper original.
    for part in windows.parts[1:] if windows.drive else windows.parts:
        if ":" in part:
            raise _ValidationError("invalid_path")


def _old_files_root(old_data_dir: Path) -> Path:
    raw = str(old_data_dir)
    _check_spelling(raw)
    root = Path(raw)
    if not root.is_absolute():
        raise _ValidationError("old_data_directory_not_absolute")
    return root / "files"


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _validate_file(raw: str, digest: str, root: Path, files: Path, old_files: Path) -> tuple[str, str]:
    _check_spelling(raw)
    path = Path(raw)
    windows = PureWindowsPath(raw)
    if path.is_absolute():
        try:
            relative = path.relative_to(old_files)
        except ValueError:
            raise _ValidationError("old_path_outside_expected_directory") from None
        if not relative.parts:
            raise _ValidationError("old_path_not_file")
        # lexists also catches dangling symlinks at the old location.
        if os.path.lexists(path):
            raise _ValidationError("old_file_still_exists")
        candidate = files / relative
        status = "repairable"
    else:
        if windows.drive or windows.root or "\\" in raw:
            raise _ValidationError("invalid_portable_path")
        if not path.parts or path.parts[0] != "files" or len(path.parts) < 2:
            raise _ValidationError("portable_path_outside_files")
        relative = Path(*path.parts[1:])
        candidate = root / path
        status = "already_portable"
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        raise _ValidationError("current_file_unavailable") from None
    if not resolved.is_relative_to(files):
        raise _ValidationError("current_path_outside_files")
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise _ValidationError("current_path_not_regular_file")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise _ValidationError("invalid_stored_sha256")
    if _digest(resolved) != digest.lower():
        raise _ValidationError("sha256_mismatch")
    portable = (Path("files") / relative).as_posix()
    return status, portable


def _validate_rows(rows: list[tuple[str, str, str]], root: Path, old_files: Path) -> tuple[list[dict], list[str]]:
    try:
        files = (root / "files").resolve(strict=True)
    except (OSError, RuntimeError):
        raise _ValidationError("files_directory_unavailable") from None
    if not files.is_dir() or not files.is_relative_to(root):
        raise _ValidationError("files_directory_outside_data")
    report_rows, portable_paths = [], []
    for index, (_, digest, raw) in enumerate(rows, start=1):
        try:
            status, portable = _validate_file(raw, digest, root, files, old_files)
            report_rows.append({"index": index, "status": status, "reason": "verified"})
            portable_paths.append(portable)
        except _ValidationError as exc:
            report_rows.append({"index": index, "status": "failed", "reason": str(exc)})
            portable_paths.append("")
        except (OSError, RuntimeError, ValueError):
            report_rows.append({"index": index, "status": "failed", "reason": "current_file_unavailable"})
            portable_paths.append("")
    return report_rows, portable_paths


def _counts(report: dict, rows: list[dict], expected_count: int | None) -> None:
    report["rows"] = rows
    report["total"] = len(rows)
    for status in ("repairable", "already_portable", "failed"):
        report[status] = sum(row["status"] == status for row in rows)
    if expected_count is not None and len(rows) != expected_count:
        report["errors"].append("record_count_mismatch")
    report["ready"] = not report["failed"] and not report["errors"]


def migrate_paper_paths(
    data_dir: Path,
    old_data_dir: Path,
    *,
    apply: bool = False,
    expected_count: int | None = 8,
    backup_dir: Path | None = None,
) -> dict:
    """Back up, verify every original, then optionally migrate in one transaction.

    ``ready`` is true only when every row and the expected count pass. A failure
    returns anonymous error codes and leaves all source rows unchanged. Relative
    ``files/...`` paths are reverified, making successful reruns safe.
    """
    report = {
        "mode": "apply" if apply else "dry-run", "total": 0,
        "repairable": 0, "already_portable": 0, "failed": 0,
        "updated": 0, "backup": None, "rows": [], "errors": [],
        "ready": False, "applied": False,
    }
    try:
        if expected_count is not None and (isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 0):
            raise _ValidationError("invalid_expected_count")
        root = Path(data_dir).resolve(strict=True)
        if not root.is_dir():
            raise _ValidationError("data_directory_unavailable")
        database = (root / DATABASE_NAME).resolve(strict=True)
        if not database.is_file() or not database.is_relative_to(root):
            raise _ValidationError("source_database_unavailable")
        # No source write connection is opened until this backup is verified.
        directory = Path(backup_dir).resolve() if backup_dir is not None else root / "backups"
        report["backup"] = str(_backup(database, directory))
        old_files = _old_files_root(old_data_dir)
        db = _connection(database)
        try:
            original_rows = _read_rows(db)
        finally:
            db.close()
        checked, portable_paths = _validate_rows(original_rows, root, old_files)
        _counts(report, checked, expected_count)
        if not apply or not report["ready"]:
            return report

        db = _connection(database, writable=True)
        try:
            db.execute("BEGIN IMMEDIATE")
            current_rows = _read_rows(db)
            # Repeat filesystem/SHA checks while holding the SQLite write lock.
            checked_again, current_paths = _validate_rows(current_rows, root, old_files)
            if any(row["status"] == "failed" for row in checked_again):
                report["errors"].clear()
                _counts(report, checked_again, expected_count)
                db.rollback()
                return report
            if current_rows != original_rows or current_paths != portable_paths:
                raise _ValidationError("database_changed_during_validation")
            updated = 0
            for row, checked_row, portable in zip(current_rows, checked_again, current_paths):
                if checked_row["status"] != "repairable":
                    continue
                paper_id, digest, old_path = row
                changed = db.execute(
                    "UPDATE papers SET file_path=? WHERE id=? AND sha256=? AND file_path=?",
                    (portable, paper_id, digest, old_path),
                ).rowcount
                if changed != 1:
                    raise _ValidationError("conditional_update_failed")
                updated += changed
            db.commit()
            report["updated"] = updated
            report["applied"] = True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
    except _ValidationError as exc:
        report["errors"].append(str(exc))
        report["ready"] = False
    except sqlite3.Error:
        report["errors"].append("database_operation_failed")
        report["ready"] = False
    except (OSError, RuntimeError, ValueError, TypeError):
        report["errors"].append("filesystem_operation_failed")
        report["ready"] = False
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--old-data-dir", required=True, type=Path)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--expected-count", type=int, default=8)
    parser.add_argument("--apply", action="store_true", help="Apply only after every validation succeeds")
    args = parser.parse_args(argv)
    report = migrate_paper_paths(
        args.data_dir, args.old_data_dir, apply=args.apply,
        expected_count=args.expected_count, backup_dir=args.backup_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
