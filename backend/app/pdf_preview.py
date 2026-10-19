"""Bounded local rendering for browsers without a native PDF viewer."""
from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from .errors import DocumentError


def _render_pdf_page_local(content: bytes, page_number: int) -> bytes:
    import pymupdf

    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.needs_pass or not 1 <= page_number <= len(document):
                raise DocumentError("文献加密或预览页码超出范围，请检查原始文件。")
            page = document[page_number - 1]
            width, height = page.rect.width, page.rect.height
            if not all(math.isfinite(value) and value > 0 for value in (width, height)):
                raise DocumentError("原文页面尺寸无效，无法生成预览。")
            scale = min(2.0, 1600.0 / max(width, height))
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False,
                                     colorspace=pymupdf.csRGB)
            return pixmap.tobytes("png")
    except DocumentError:
        raise
    except Exception:
        raise DocumentError("原文页面预览生成失败，请使用原始文件入口检查文献。") from None


def render_pdf_page(content: bytes, page_number: int) -> bytes:
    """Run MuPDF in an isolated process, never in the API parsing process.

    MuPDF does not support concurrent use by multiple application threads. A
    private one-shot worker also lets a timed-out render be terminated without
    killing the API or leaving its native renderer running in a thread.
    """
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    worker = Path(__file__).with_name("pdf_preview_worker.py")
    # A file handle avoids communicate(input=...) synchronously writing a pipe
    # before its timeout begins on Windows. This private temporary copy is never
    # registered in the research database and is removed when its handles close.
    with tempfile.TemporaryFile() as source:
        source.write(content)
        source.seek(0)
        process = subprocess.Popen([sys.executable, "-B", str(worker), str(page_number)], stdin=source,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **options)
        pipes_drained = False
        try:
            try:
                output, _ = process.communicate(timeout=20)
                pipes_drained = True
            except subprocess.TimeoutExpired:
                # Windows virtualenv launchers may have a child interpreter;
                # target only the tree belonging to this temporary worker.
                try:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=5, check=False)
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                try:
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=5)
                    pipes_drained = True
                except (OSError, subprocess.TimeoutExpired):
                    # Do not enter Popen.__exit__ or close a pipe while its
                    # daemon reader is stuck: both can wait without a bound.
                    # OS termination failure may leave that worker/reader alive;
                    # it must not also hold this API request indefinitely.
                    pass
                raise DocumentError("原文页面预览超时，请使用完整原始文件入口。") from None
            if process.returncode == 3:
                raise ImportError("PDF preview component unavailable")
            if process.returncode != 0 or not output.startswith(b"\x89PNG\r\n\x1a\n") or len(output) > 12 * 1024 * 1024:
                raise DocumentError("原文页面预览生成失败，请使用完整原始文件入口。")
            return output
        finally:
            if pipes_drained and process.stdout is not None:
                process.stdout.close()
