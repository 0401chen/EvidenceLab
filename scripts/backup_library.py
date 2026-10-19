"""Create a portable, verified library backup in a NEW folder; never restore over data."""
from __future__ import annotations
import argparse, hashlib, json, sqlite3
from pathlib import Path, PureWindowsPath

def backup(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Backup destination must be a new folder outside the source library")
    database = source / "evidencelab.sqlite3"
    if not database.is_file() or not (source/"files").resolve().is_relative_to(source):
        raise ValueError("Source database is missing")
    output.mkdir(parents=True)
    # A failed/incomplete backup never has a completion manifest. Keep it for
    # diagnosis rather than deleting anything or claiming it is recoverable.
    with sqlite3.connect(database.as_uri()+"?mode=ro", uri=True) as src:
        with sqlite3.connect(output/database.name) as dst:
            src.backup(dst)
    with sqlite3.connect((output/database.name).as_uri()+"?mode=ro", uri=True) as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Database integrity check failed")
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        counts = {t:db.execute('SELECT count(*) FROM "'+t.replace('"','""')+'"').fetchone()[0] for t in tables}
        rows = db.execute("SELECT sha256,file_path FROM papers").fetchall()
    files = {}
    for digest, stored in rows:
        if not stored:
            continue
        relative = Path(stored)
        path = (source/relative).resolve()
        if (relative.is_absolute() or PureWindowsPath(stored).is_absolute() or PureWindowsPath(stored).drive
                or ".." in relative.parts or not path.is_relative_to((source/"files").resolve())
                or not path.is_file()):
            raise ValueError("Original file does not satisfy portable library boundaries")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Original file SHA256 mismatch; backup is incomplete")
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write(content)
        files[relative.as_posix()] = digest
    manifest = {"complete":True,"schema":"portable_library_backup_v1","table_counts":counts,"files":files,
                "database_sha256":hashlib.sha256((output/database.name).read_bytes()).hexdigest(),
                "restore":"Use a copy as a separate data directory; never overwrite a running library."}
    (output/"backup-complete.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf8")
    return {"complete":True,"papers":counts.get("papers",0),"original_files":len(files)}

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    try:
        print(json.dumps(backup(args.source,args.output)))
    except (ValueError,OSError,sqlite3.Error):
        raise SystemExit("Backup failed or incomplete. Source unchanged; do not use output without backup-complete.json.")
