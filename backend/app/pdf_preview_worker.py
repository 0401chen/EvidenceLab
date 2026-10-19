"""One PDF render per process; binary source enters stdin, PNG leaves stdout."""
from contextlib import redirect_stdout
from pathlib import Path
import sys


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.app.pdf_preview import _render_pdf_page_local

    content = sys.stdin.buffer.read(32 * 1024 * 1024 + 1)
    if len(content) > 32 * 1024 * 1024:
        return 2
    binary_output = sys.stdout.buffer
    try:
        with redirect_stdout(sys.stderr):
            result = _render_pdf_page_local(content, int(sys.argv[1]))
    except ImportError:
        return 3
    except Exception:
        return 2
    binary_output.write(result)
    binary_output.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
