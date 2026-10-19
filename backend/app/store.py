from __future__ import annotations

import json
import hashlib
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class Store:
    """Small local, transactional repository. One connection per operation."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.files = self.root / "files"
        self.files.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "evidencelab.sqlite3"
        self.lock = threading.RLock()
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY, sha256 TEXT UNIQUE NOT NULL,
                    payload TEXT NOT NULL, file_path TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    id TEXT PRIMARY KEY, paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_evidence_paper ON evidence(paper_id);
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY, paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_experiments_paper ON experiments(paper_id);
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS extraction_candidates (
                    id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_candidates_paper ON extraction_candidates(paper_id);
                CREATE TABLE IF NOT EXISTS experiment_history (
                    id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                    experiment_id TEXT NOT NULL, payload TEXT NOT NULL,
                    created_at TEXT NOT NULL, reason TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS report_states (
                    report_id TEXT PRIMARY KEY REFERENCES reports(id) ON DELETE CASCADE,
                    reason TEXT NOT NULL, changed_at TEXT NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def list_papers(self) -> list[dict]:
        with self.connect() as db:
            papers = [json.loads(row[0]) for row in db.execute("SELECT payload FROM papers ORDER BY rowid DESC")]
        return papers

    def find_by_hash(self, digest: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM papers WHERE sha256=?", (digest,)).fetchone()
        return json.loads(row[0]) if row else None

    def get_paper(self, paper_id: str) -> dict | None:
        with self.connect() as db:
            db.execute("BEGIN")
            row = db.execute("SELECT payload FROM papers WHERE id=?", (paper_id,)).fetchone()
            if not row:
                return None
            result = json.loads(row[0])
            result["evidence"] = [json.loads(x[0]) for x in db.execute("SELECT payload FROM evidence WHERE paper_id=? ORDER BY rowid", (paper_id,))]
            result["experiments"] = [json.loads(x[0]) for x in db.execute("SELECT payload FROM experiments WHERE paper_id=? ORDER BY rowid", (paper_id,))]
            result["experiment_revisions"] = {item["id"]: self.experiment_payload_revision(item) for item in result["experiments"]}
            result["extraction_candidates"] = self._candidates(db, paper_id, result["experiments"])
        return result

    def save_paper(self, paper: dict, digest: str, file_path: Path, evidence: list[dict], experiments: list[dict]):
        path = self._resolve_file_path(file_path)
        # Anchor persisted paths to the data directory, so moving it preserves links.
        portable = (Path("files") / path.relative_to(self.files.resolve())).as_posix()
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO papers VALUES(?,?,?,?)", (paper["id"], digest, dumps(paper), portable))
            db.executemany("INSERT INTO evidence VALUES(?,?,?)", [(e["id"], paper["id"], dumps(e)) for e in evidence])
            db.executemany("INSERT INTO experiments VALUES(?,?,?)", [(e["id"], paper["id"], dumps(e)) for e in experiments])

    def get_file(self, paper_id: str) -> Path | None:
        with self.connect() as db:
            row = db.execute("SELECT file_path FROM papers WHERE id=?", (paper_id,)).fetchone()
        if not row:
            return None
        return self._resolve_file_path(row[0])

    def get_file_hash(self, paper_id: str) -> str | None:
        with self.connect() as db:
            row = db.execute("SELECT sha256 FROM papers WHERE id=?", (paper_id,)).fetchone()
        return row[0] if row else None

    def _resolve_file_path(self, value: str | Path) -> Path:
        raw = str(value)
        path = Path(raw)
        windows = PureWindowsPath(raw)
        # Check both separator styles, including Windows drive/root-relative paths.
        if not raw or "\x00" in raw or ".." in path.parts or ".." in windows.parts:
            raise ValueError("文件路径超出文献目录")
        if not path.is_absolute() and (windows.drive or windows.root):
            raise ValueError("文件路径超出文献目录")
        candidate = path if path.is_absolute() else self.root / path
        files = self.files.resolve()
        resolved = candidate.resolve()
        if not files.is_relative_to(self.root) or not resolved.is_relative_to(files):
            raise ValueError("文件路径超出文献目录")
        return resolved

    def evidence_for(self, paper_ids: list[str]) -> list[dict]:
        if not paper_ids:
            return []
        marks = ",".join("?" for _ in paper_ids)
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(f"SELECT payload FROM evidence WHERE paper_id IN ({marks}) ORDER BY rowid", paper_ids)]

    def experiments_for(self, paper_ids: list[str]) -> list[dict]:
        if not paper_ids:
            return []
        marks = ",".join("?" for _ in paper_ids)
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(f"SELECT payload FROM experiments WHERE paper_id IN ({marks}) ORDER BY rowid", paper_ids)]

    def get_experiment(self, experiment_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def replace_experiments(self, paper_id: str, experiments: list[dict]):
        with self.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM papers WHERE id=?", (paper_id,)).fetchone()
            if not row:
                raise KeyError(paper_id)
            paper = json.loads(row[0])
            self._archive_experiments(db, paper_id, "replacement")
            paper["experiment_count"] = len(experiments)
            db.execute("DELETE FROM experiments WHERE paper_id=?", (paper_id,))
            db.executemany("INSERT INTO experiments VALUES(?,?,?)", [(e["id"], paper_id, dumps(e)) for e in experiments])
            db.execute("UPDATE papers SET payload=? WHERE id=?", (dumps(paper), paper_id))
            self._invalidate_reports(db, paper_id)

    def update_experiment(self, experiment: dict, expected: dict | None = None, expected_revision: str | None = None) -> bool:
        with self.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM experiments WHERE id=?", (experiment["id"],)).fetchone()
            if not row:
                return False
            current = json.loads(row[0])
            if ((expected is not None and current != expected)
                    or (expected_revision is not None and self.experiment_payload_revision(current) != expected_revision)):
                return False
            self._archive_experiments(db, experiment["paper_id"], "manual_edit", experiment["id"])
            db.execute("UPDATE experiments SET payload=? WHERE id=?", (dumps(experiment), experiment["id"]))
            self._invalidate_reports(db, experiment["paper_id"])
            return True

    @staticmethod
    def experiment_payload_revision(experiment: dict) -> str:
        """A stable token for the stored record; never inserted into its payload."""
        canonical = json.dumps(experiment, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def experiment_revision(experiments: list[dict]) -> str:
        ordered = sorted(experiments, key=lambda item: item["id"])
        return hashlib.sha256(json.dumps(ordered, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()

    @staticmethod
    def _experiment_identity(experiment: dict) -> str:
        # Names, IDs, extraction timestamps and notes do not define a new result.
        # Source IDs prevent merging equal measurements from separate experiments.
        identity = {key: experiment.get(key) for key in
                    ("paper_id", "task", "method", "dataset", "split", "metric", "value", "unit", "sample_size", "conditions")}
        identity["evidence_ids"] = sorted(set(experiment.get("evidence_ids", [])))
        return json.dumps(identity, ensure_ascii=False, sort_keys=True, allow_nan=False)

    @staticmethod
    def _archive_experiments(db, paper_id: str, reason: str, experiment_id: str | None = None):
        query = "SELECT id,payload FROM experiments WHERE paper_id=?"
        params = [paper_id]
        if experiment_id is not None:
            query += " AND id=?"
            params.append(experiment_id)
        timestamp = datetime.now(timezone.utc).isoformat()
        rows = db.execute(query, params).fetchall()
        db.executemany("INSERT INTO experiment_history VALUES(?,?,?,?,?,?)",
                       [(uuid.uuid4().hex, paper_id, row["id"], row["payload"], timestamp, reason) for row in rows])

    def experiment_history(self, experiment_id: str) -> list[dict]:
        with self.connect() as db:
            return [{"id": row["id"], "experiment": json.loads(row["payload"]), "created_at": row["created_at"], "reason": row["reason"]}
                    for row in db.execute("SELECT * FROM experiment_history WHERE experiment_id=? ORDER BY rowid DESC", (experiment_id,))]

    def save_extraction_candidate(self, paper_id: str, experiments: list[dict], mode: str, base_revision: str,
                                  source_evidence: list[dict] | None = None, parser: str | None = None,
                                  extraction_scope: dict | None = None) -> dict:
        candidate = {"id": uuid.uuid4().hex, "paper_id": paper_id, "mode": mode, "status": "pending",
                     "created_at": datetime.now(timezone.utc).isoformat(), "base_revision": base_revision,
                     "experiments": [{**item, "id": f"exp_{uuid.uuid4().hex}"} for item in experiments]}
        if source_evidence is not None:
            candidate.update(reparsed=True, source_evidence=source_evidence, parser=parser)
        if extraction_scope is not None:
            candidate["extraction_scope"] = extraction_scope
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO extraction_candidates VALUES(?,?,?)", (candidate["id"], paper_id, dumps(candidate)))
        return candidate

    def _candidates(self, db, paper_id: str, experiments: list[dict]) -> list[dict]:
        revision = self.experiment_revision(experiments)
        candidates = [json.loads(row[0]) for row in db.execute(
            "SELECT payload FROM extraction_candidates WHERE paper_id=? ORDER BY rowid DESC", (paper_id,))]
        for candidate in candidates:
            candidate["stale"] = candidate["status"] == "pending" and candidate["base_revision"] != revision
        return candidates

    def accept_extraction_candidate(self, paper_id: str, candidate_id: str, selected_ids: list[str] | None = None) -> dict:
        """Append explicitly selected results atomically, preserving all existing edits."""
        with self.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM extraction_candidates WHERE id=? AND paper_id=?", (candidate_id, paper_id)).fetchone()
            if not row:
                raise KeyError(candidate_id)
            candidate = json.loads(row[0])
            if candidate["status"] == "accepted":
                return candidate  # Double clicks and request retries are idempotent.
            if candidate["status"] != "pending":
                raise ValueError("候选结果已被放弃，请重新提取。")
            current = [json.loads(row[0]) for row in db.execute("SELECT payload FROM experiments WHERE paper_id=?", (paper_id,))]
            if self.experiment_revision(current) != candidate["base_revision"]:
                raise ValueError("实验记录已在提取后更新。为保护人工修正，请重新提取候选后再采纳。")
            known = {item["id"]: item for item in candidate["experiments"]}
            selected = list(known) if selected_ids is None else list(dict.fromkeys(selected_ids))
            if not selected or not set(selected).issubset(known):
                raise ValueError("请至少选择一条属于当前候选的实验。")
            identities = {self._experiment_identity(item): item["id"] for item in current}
            existing_evidence = {row["id"]: json.loads(row["payload"]) for row in db.execute("SELECT id,payload FROM evidence WHERE paper_id=?", (paper_id,))}
            added_evidence = []
            for source in candidate.get("source_evidence", []):
                if source.get("paper_id") != paper_id:
                    raise ValueError("候选原文不属于当前论文，请重新解析。")
                previous = existing_evidence.get(source["id"])
                if previous is not None and previous != source:
                    raise ValueError("原文在候选生成后已更新，请重新解析。")
                if previous is None:
                    db.execute("INSERT INTO evidence VALUES(?,?,?)", (source["id"], paper_id, dumps(source)))
                    existing_evidence[source["id"]] = source
                    added_evidence.append(source["id"])
            accepted, inserted = [], []
            for identifier in selected:
                item = known[identifier]
                identity = self._experiment_identity(item)
                if identity not in identities:
                    db.execute("INSERT INTO experiments VALUES(?,?,?)", (item["id"], paper_id, dumps(item)))
                    identities[identity] = item["id"]
                    inserted.append(item["id"])
                accepted.append(identities[identity])
            paper_row = db.execute("SELECT payload FROM papers WHERE id=?", (paper_id,)).fetchone()
            if not paper_row:
                raise KeyError(paper_id)
            if inserted or added_evidence:
                paper = json.loads(paper_row[0])
                paper["experiment_count"] = len(current) + len(inserted)
                paper["evidence_count"] = len(existing_evidence)
                db.execute("UPDATE papers SET payload=? WHERE id=?", (dumps(paper), paper_id))
                self._invalidate_reports(db, paper_id)
            candidate.update(status="accepted", accepted_at=datetime.now(timezone.utc).isoformat(),
                             accepted_experiment_ids=list(dict.fromkeys(accepted)), inserted_experiment_ids=inserted,
                             inserted_evidence_ids=added_evidence)
            db.execute("UPDATE extraction_candidates SET payload=? WHERE id=?", (dumps(candidate), candidate_id))
            return candidate

    def reject_extraction_candidate(self, paper_id: str, candidate_id: str):
        with self.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM extraction_candidates WHERE id=? AND paper_id=?", (candidate_id, paper_id)).fetchone()
            if not row:
                raise KeyError(candidate_id)
            candidate = json.loads(row[0])
            if candidate["status"] == "accepted":
                raise ValueError("候选已采纳，无法再放弃；已有实验记录已保留。")
            candidate["status"] = "rejected"
            db.execute("UPDATE extraction_candidates SET payload=? WHERE id=?", (dumps(candidate), candidate_id))

    @staticmethod
    def _invalidate_reports(db, paper_id):
        for row in db.execute("SELECT id,payload FROM reports").fetchall():
            report = json.loads(row["payload"])
            if any(p["id"] == paper_id for p in report.get("papers", [])):
                # Keep the immutable research snapshot. Stale exports require an
                # explicit opt-in so old results cannot look like current data.
                db.execute("INSERT OR IGNORE INTO report_states VALUES(?,?,?)",
                           (row["id"], "相关文献或实验已更新；这是历史快照，请重新分析获取当前结果。", datetime.now(timezone.utc).isoformat()))

    def delete_paper(self, paper_id: str) -> bool:
        with self.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._invalidate_reports(db, paper_id)
            return bool(db.execute("DELETE FROM papers WHERE id=?", (paper_id,)).rowcount)

    def save_report(self, report: dict) -> bool:
        with self.lock, self.connect() as db:
            # Validate and save one snapshot in the same write transaction.
            db.execute("BEGIN IMMEDIATE")
            paper_ids = [paper["id"] for paper in report["papers"]]
            if not paper_ids:
                return False
            marks = ",".join("?" for _ in paper_ids)
            present = db.execute(f"SELECT COUNT(*) FROM papers WHERE id IN ({marks})", paper_ids).fetchone()[0]
            current = {row["id"]: json.loads(row["payload"]) for row in db.execute(
                f"SELECT id,payload FROM experiments WHERE paper_id IN ({marks})", paper_ids)}
            expected = {item["id"]: item for item in report["experiments"]}
            current_evidence = {row["id"]: json.loads(row["payload"]) for row in db.execute(
                f"SELECT id,payload FROM evidence WHERE paper_id IN ({marks})", paper_ids)}
            expected_evidence = {item["id"]: item for item in report.get("evidence", [])}
            if present != len(paper_ids) or current != expected or current_evidence != expected_evidence:
                return False
            db.execute("INSERT INTO reports VALUES(?,?)", (report["id"], dumps(report)))
            return True

    def get_report(self, report_id: str, include_stale: bool = False) -> dict | None:
        with self.connect() as db:
            db.execute("BEGIN")
            row = db.execute("SELECT payload FROM reports WHERE id=?", (report_id,)).fetchone()
            state = db.execute("SELECT reason,changed_at FROM report_states WHERE report_id=?", (report_id,)).fetchone()
        if not row or (state and not include_stale):
            return None
        result = json.loads(row[0])
        if state:
            result.update(stale=True, stale_reason=state["reason"], stale_since=state["changed_at"])
        return result

    def list_reports(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT r.payload,s.reason FROM reports r LEFT JOIN report_states s ON s.report_id=r.id ORDER BY r.rowid DESC").fetchall()
        result = []
        for row in rows:
            report = json.loads(row["payload"])
            result.append({"id": report["id"], "created_at": report["created_at"], "papers": report.get("papers", []),
                           "experiment_count": len(report.get("experiments", [])), "comparison_count": len(report.get("comparisons", [])),
                           "stale": row["reason"] is not None, "stale_reason": row["reason"]})
        return result
