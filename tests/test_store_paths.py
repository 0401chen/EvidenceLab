"""Portable original-file paths and atomic, offline relocation recovery."""

from __future__ import annotations

import hashlib
import json
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import subprocess

import pytest

from backend.app.store import Store


def database_snapshot(database):
    """Keep complete rows, so preservation covers payloads as well as counts."""
    with closing(sqlite3.connect(database)) as connection:
        return {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
            for table in ("papers", "evidence", "experiments", "reports")
        }


def add_paper(store, index=0):
    paper_id = f"paper-private-{index}"
    content = f"# Private synthetic source {index}\n\nMeasured efficiency: {80 + index}%.".encode()
    digest = hashlib.sha256(content).hexdigest()
    file_path = store.files / f"private-source-{index}.md"
    file_path.write_bytes(content)
    paper = {"id": paper_id, "title": f"private-title-{index}", "filename": file_path.name,
             "status": "ready", "is_demo": False, "page_count": 1,
             "evidence_count": 1, "experiment_count": 1}
    evidence = {"id": f"evidence-private-{index}", "paper_id": paper_id,
                "page": 1, "text": "private synthetic evidence"}
    experiment = {"id": f"experiment-private-{index}", "paper_id": paper_id,
                  "name": "private synthetic experiment", "value": 80 + index,
                  "evidence_ids": [evidence["id"]]}
    store.save_paper(paper, digest, file_path, [evidence], [experiment])
    return paper, file_path, content, digest


@pytest.fixture
def relocated_legacy(tmp_path):
    old_project = tmp_path / "a黑客松"
    old_data = old_project / "data"
    store = Store(old_data)
    sources = [add_paper(store, index) for index in range(8)]
    with store.connect() as connection:
        for paper, path, _, _ in sources:
            connection.execute("UPDATE papers SET file_path=? WHERE id=?", (str(path), paper["id"]))
        connection.execute("INSERT INTO reports VALUES (?,?)", (
            "report-private", json.dumps({"id": "report-private", "papers": [s[0] for s in sources],
                                           "private_notes": "preserve this exact saved report"}),
        ))
    new_project = tmp_path / "aheikesong"
    old_project.rename(new_project)
    current = new_project / "data"
    assert not old_project.exists()
    return {"old": old_data, "current": current, "database": current / store.database.name,
            "sources": sources, "backup_dir": tmp_path / "backups"}


def migrate(workspace, *, apply=False, expected_count=8):
    from scripts.migrate_paper_paths import migrate_paper_paths

    return migrate_paper_paths(workspace["current"], workspace["old"], apply=apply,
                               expected_count=expected_count, backup_dir=workspace["backup_dir"])


def test_save_paper_persists_a_data_relative_path(tmp_path):
    store = Store(tmp_path / "data")
    paper, path, content, _ = add_paper(store)
    with store.connect() as connection:
        saved = connection.execute("SELECT file_path FROM papers WHERE id=?", (paper["id"],)).fetchone()[0]
    assert saved == f"files/{path.name}"
    assert store.get_file(paper["id"]).read_bytes() == content


def test_project_rename_keeps_original_file_available(tmp_path):
    project = tmp_path / "a黑客松"
    store = Store(project / "data")
    paper, path, content, _ = add_paper(store)
    relocated = tmp_path / "aheikesong"
    project.rename(relocated)
    reopened = Store(relocated / "data")
    assert reopened.get_file(paper["id"]) == (reopened.files / path.name).resolve()
    assert reopened.get_file(paper["id"]).read_bytes() == content


def test_current_absolute_legacy_path_remains_supported(tmp_path):
    store = Store(tmp_path / "data")
    paper, path, content, _ = add_paper(store)
    with store.connect() as connection:
        connection.execute("UPDATE papers SET file_path=? WHERE id=?", (str(path), paper["id"]))
    assert store.get_file(paper["id"]).read_bytes() == content


@pytest.mark.parametrize("value", [
    "../files/private-source-0.md", "files/../files/private-source-0.md",
    "files/../../outside.md", r"files\..\..\outside.md",
    r"files/..\private-source-0.md", r"D:files\private-source-0.md",
    r"\files\private-source-0.md", "", "files/invalid\x00.md",
])
def test_store_rejects_traversal_and_ambiguous_paths_on_read_and_save(tmp_path, value):
    store = Store(tmp_path / "data")
    paper, _, _, digest = add_paper(store)
    with store.connect() as connection:
        connection.execute("UPDATE papers SET file_path=? WHERE id=?", (value, paper["id"]))
    with pytest.raises(ValueError):
        store.get_file(paper["id"])
    with pytest.raises(ValueError):
        store.save_paper({"id": "rejected-paper"}, digest + "x", value, [], [])
    assert store.get_paper("rejected-paper") is None


@pytest.mark.parametrize("outside_name", ["outside.md", "files-extra/private-source-0.md"])
def test_store_rejects_absolute_paths_outside_files(tmp_path, outside_name):
    store = Store(tmp_path / "data")
    paper, _, _, digest = add_paper(store)
    outside = store.root / outside_name
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("outside the original-file boundary")
    with store.connect() as connection:
        connection.execute("UPDATE papers SET file_path=? WHERE id=?", (str(outside), paper["id"]))
    with pytest.raises(ValueError):
        store.get_file(paper["id"])
    with pytest.raises(ValueError):
        store.save_paper({"id": "rejected-paper"}, digest + "x", outside, [], [])
    assert outside.read_text() == "outside the original-file boundary"


def create_directory_link_or_skip(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        # Windows directory junctions exercise the same resolve/boundary check
        # without requiring the privilege needed for native symbolic links.
        if os.name == "nt":
            result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                    capture_output=True, check=False)
            if result.returncode == 0:
                return
        pytest.skip(f"Symlinks are unavailable on this test host: {type(exc).__name__}")


def test_store_rejects_a_symlink_escaping_files(tmp_path):
    store = Store(tmp_path / "data")
    paper, _, _, digest = add_paper(store)
    external = tmp_path / "external"
    external.mkdir()
    outside = external / "outside.md"
    outside.write_bytes(b"synthetic external content")
    link = store.files / "escape"
    create_directory_link_or_skip(link, external)
    with store.connect() as connection:
        connection.execute("UPDATE papers SET file_path=? WHERE id=?", ("files/escape/outside.md", paper["id"]))
    with pytest.raises(ValueError):
        store.get_file(paper["id"])
    with pytest.raises(ValueError):
        store.save_paper({"id": "rejected-paper"}, digest + "x", link / outside.name, [], [])
    assert outside.read_bytes() == b"synthetic external content"


def test_migration_dry_run_is_private_and_does_not_update_rows(relocated_legacy):
    workspace = relocated_legacy
    before = database_snapshot(workspace["database"])
    report = migrate(workspace)
    assert report["mode"] == "dry-run"
    assert report["total"] == report["repairable"] == 8
    assert report["failed"] == report["updated"] == 0
    assert report["applied"] is False
    assert len(report["rows"]) == 8
    rendered = json.dumps(report, ensure_ascii=False)
    for paper, path, _, digest in workspace["sources"]:
        for private_value in (paper["id"], paper["title"], path.name, digest):
            assert private_value not in rendered
    assert database_snapshot(workspace["database"]) == before


def test_migration_backs_up_and_only_changes_paths(relocated_legacy):
    workspace = relocated_legacy
    before = database_snapshot(workspace["database"])
    report = migrate(workspace, apply=True)
    assert report["applied"] is True
    assert report["updated"] == report["repairable"] == 8
    assert report["failed"] == 0
    backup = Path(report["backup"])
    assert backup.is_file()
    assert backup.resolve().is_relative_to(workspace["backup_dir"].resolve())
    assert database_snapshot(backup) == before
    after = database_snapshot(workspace["database"])
    assert {key: rows for key, rows in after.items() if key != "papers"} == {
        key: rows for key, rows in before.items() if key != "papers"
    }
    for old_row, new_row in zip(before["papers"], after["papers"], strict=True):
        assert new_row[:3] == old_row[:3]
        assert new_row[3] == f"files/{Path(old_row[3]).name}"
    store = Store(workspace["current"])
    for paper, _, content, _ in workspace["sources"]:
        assert store.get_file(paper["id"]).read_bytes() == content
    repeated = migrate(workspace, apply=True)
    assert repeated["failed"] == repeated["updated"] == repeated["repairable"] == 0
    assert repeated["already_portable"] == 8
    assert database_snapshot(workspace["database"]) == after


def test_one_sha256_mismatch_blocks_all_migration_updates(relocated_legacy):
    workspace = relocated_legacy
    _, old_path, _, _ = workspace["sources"][-1]
    (workspace["current"] / "files" / old_path.name).write_bytes(b"wrong synthetic content")
    before = database_snapshot(workspace["database"])
    report = migrate(workspace, apply=True)
    assert report["repairable"] == 7
    assert report["failed"] == 1
    assert report["applied"] is False
    assert report["updated"] == 0
    assert database_snapshot(workspace["database"]) == before


@pytest.mark.parametrize("invalid_kind", ["outside_old_files", "old_traversal", "missing_new_file"])
def test_migration_rejects_untrusted_or_missing_sources(relocated_legacy, invalid_kind):
    workspace = relocated_legacy
    paper, old_path, _, _ = workspace["sources"][-1]
    if invalid_kind == "missing_new_file":
        (workspace["current"] / "files" / old_path.name).unlink()
    else:
        value = (workspace["old"] / "files-extra" / old_path.name if invalid_kind == "outside_old_files"
                 else workspace["old"] / "files" / ".." / "files" / old_path.name)
        with closing(sqlite3.connect(workspace["database"])) as connection, connection:
            connection.execute("UPDATE papers SET file_path=? WHERE id=?", (str(value), paper["id"]))
    before = database_snapshot(workspace["database"])
    report = migrate(workspace, apply=True)
    assert report["failed"] == 1
    assert report["updated"] == 0
    assert report["applied"] is False
    assert database_snapshot(workspace["database"]) == before


def test_migration_rejects_a_new_file_symlink_outside_files(relocated_legacy, tmp_path):
    workspace = relocated_legacy
    paper, old_path, content, _ = workspace["sources"][-1]
    external = tmp_path / "external"
    external.mkdir()
    outside = external / old_path.name
    outside.write_bytes(content)  # Matching bytes cannot override the directory boundary.
    create_directory_link_or_skip(workspace["current"] / "files" / "escape", external)
    with closing(sqlite3.connect(workspace["database"])) as connection, connection:
        connection.execute("UPDATE papers SET file_path=? WHERE id=?", (
            str(workspace["old"] / "files" / "escape" / old_path.name), paper["id"],
        ))
    before = database_snapshot(workspace["database"])
    report = migrate(workspace, apply=True)
    assert report["failed"] == 1
    assert report["updated"] == 0
    assert report["applied"] is False
    assert database_snapshot(workspace["database"]) == before
    assert outside.read_bytes() == content


def test_migration_refuses_an_unexpected_paper_count(relocated_legacy):
    workspace = relocated_legacy
    before = database_snapshot(workspace["database"])
    report = migrate(workspace, apply=True, expected_count=7)
    assert report["ready"] is False
    assert "record_count_mismatch" in report["errors"]
    assert report["applied"] is False
    assert report["updated"] == 0
    assert database_snapshot(workspace["database"]) == before


def test_sqlite_write_failure_rolls_back_every_path_update(relocated_legacy):
    workspace = relocated_legacy
    with closing(sqlite3.connect(workspace["database"])) as connection, connection:
        connection.execute("""CREATE TRIGGER reject_final_path_update
            BEFORE UPDATE OF file_path ON papers WHEN OLD.id = 'paper-private-7'
            BEGIN SELECT RAISE(ABORT, 'synthetic private SQL failure'); END""")
    before = database_snapshot(workspace["database"])
    report = migrate(workspace, apply=True)
    assert report["repairable"] == 8
    assert report["applied"] is False
    assert report["updated"] == 0
    assert "synthetic private SQL failure" not in json.dumps(report)
    assert database_snapshot(workspace["database"]) == before
    assert database_snapshot(Path(report["backup"])) == before


@pytest.mark.parametrize("legacy_absolute_path", [False, True])
def test_http_original_and_duplicate_upload_survive_data_relocation(client, tmp_path, legacy_absolute_path):
    from fastapi.testclient import TestClient
    from backend.app.main import create_app

    content = b"# Synthetic relocated paper\n\nCobalt catalyst efficiency is 84 percent."
    response = client.post("/api/papers/upload", files={"files": ("original.md", content, "text/markdown")})
    assert response.status_code == 200
    assert response.json()["errors"] == []
    paper_id = response.json()["papers"][0]["id"]
    store = client.app.state.store
    old_data = store.root
    if legacy_absolute_path:
        path = store.get_file(paper_id)
        with store.connect() as connection:
            connection.execute("UPDATE papers SET file_path=? WHERE id=?", (str(path), paper_id))
    before = database_snapshot(store.database)
    relocated = tmp_path / "renamed-project" / "data"
    relocated.parent.mkdir()
    old_data.rename(relocated)
    if legacy_absolute_path:
        from scripts.migrate_paper_paths import migrate_paper_paths

        report = migrate_paper_paths(relocated, old_data, apply=True, expected_count=1,
                                    backup_dir=tmp_path / "http-backup")
        assert report["applied"] is True
        assert report["updated"] == 1
    with TestClient(create_app(relocated)) as reopened:
        original = reopened.get(f"/api/papers/{paper_id}/file")
        assert original.status_code == 200
        assert original.content == content
        assert original.headers["x-content-type-options"] == "nosniff"
        duplicate = reopened.post("/api/papers/upload", files={"files": ("renamed.md", content, "text/markdown")})
        assert duplicate.status_code == 200
        assert duplicate.json()["errors"] == []
        assert duplicate.json()["papers"][0]["id"] == paper_id
        assert len(reopened.get("/api/papers").json()) == 1
        assert len(list(reopened.app.state.store.files.iterdir())) == 1
        assert reopened.get(f"/api/papers/{paper_id}/file").content == content
        after = database_snapshot(reopened.app.state.store.database)
        assert after["evidence"] == before["evidence"]
        assert after["experiments"] == before["experiments"]
        assert after["reports"] == before["reports"]
        assert after["papers"][0][:3] == before["papers"][0][:3]
