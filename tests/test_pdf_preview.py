"""Preview the real selected page, with the same file boundary and SHA guard."""
import hashlib

import pymupdf
import pytest


def upload_pdf(client):
    with pymupdf.open() as document:
        for color in ((1, 0, 0), (0, 0, 1)):
            page = document.new_page(width=400, height=300)
            page.draw_rect(page.rect, color=color, fill=color)
            page.insert_text((20, 30), "Original research page with visible source text.", color=(1, 1, 1))
        payload = document.tobytes()
    response = client.post("/api/papers/upload", files=[("files", ("two-pages.pdf", payload, "application/pdf"))])
    assert response.status_code == 200 and not response.json()["errors"]
    return response.json()["papers"][0]


def test_original_page_identity_and_image_bounds_without_research_changes(client):
    paper = upload_pdf(client)
    before = client.get(f"/api/papers/{paper['id']}").json()
    for page, expected in ((1, (255, 0, 0)), (2, (0, 0, 255))):
        response = client.get(f"/api/papers/{paper['id']}/pages/{page}/preview")
        assert response.status_code == 200 and response.headers["content-type"] == "image/png"
        image = pymupdf.Pixmap(response.content)
        assert max(image.width, image.height) <= 1600
        assert image.pixel(image.width // 2, image.height // 2) == expected
        assert "private" in response.headers["cache-control"]
    assert client.get(f"/api/papers/{paper['id']}").json() == before


@pytest.mark.parametrize("page", [0, -1, 3])
def test_preview_does_not_silently_render_a_different_page(client, page):
    paper = upload_pdf(client)
    assert client.get(f"/api/papers/{paper['id']}/pages/{page}/preview").status_code == 422


def test_preview_rejects_changed_file_and_escaping_path(client, tmp_path):
    paper = upload_pdf(client)
    store = client.app.state.store
    original = store.get_file(paper["id"])
    content = original.read_bytes()
    original.write_bytes(content + b"\nchanged")
    assert client.get(f"/api/papers/{paper['id']}/pages/1/preview").status_code == 409
    # Even a byte-identical external file must not bypass the path boundary.
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(content)
    with store.connect() as database:
        database.execute("UPDATE papers SET file_path=?,sha256=? WHERE id=?", (str(outside), hashlib.sha256(content).hexdigest(), paper["id"]))
    assert client.get(f"/api/papers/{paper['id']}/pages/1/preview").status_code == 409


def test_text_file_keeps_original_access_without_attempting_pdf_render(client):
    response = client.post("/api/papers/upload", files=[("files", ("source.md", b"# Study\n\nOriginal text.", "text/markdown"))])
    identifier = response.json()["papers"][0]["id"]
    assert client.get(f"/api/papers/{identifier}/pages/1/preview").status_code == 415
    assert client.get(f"/api/papers/{identifier}/file").status_code == 200


def test_render_workers_do_not_call_mupdf_in_the_application_process(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from backend.app.pdf_preview import render_pdf_page
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((30, 50), "Independent original page.")
        content = document.tobytes()
    # If rendering is accidentally moved back into the API's thread pool, this
    # fails. Child interpreters import their own renderer instead.
    monkeypatch.setattr(pymupdf, "open", lambda *args, **kwargs: pytest.fail("MuPDF ran in the parent process"))
    with ThreadPoolExecutor(max_workers=2) as executor:
        images = list(executor.map(lambda _: render_pdf_page(content, 1), range(2)))
    assert all(image.startswith(b"\x89PNG\r\n\x1a\n") for image in images)


def test_render_timeout_stays_controlled_when_worker_cleanup_also_times_out(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    import subprocess

    from backend.app import pdf_preview
    from backend.app.errors import DocumentError

    process = MagicMock()
    process.pid = 123456789
    process.poll.return_value = None
    # Even after kill(), the simulated launcher/child remains alive and holds
    # stdout. Neither a context-manager exit nor closing that pipe may run.
    process.communicate.side_effect = [
        subprocess.TimeoutExpired("pdf-preview-worker", 20),
        subprocess.TimeoutExpired("pdf-preview-worker", 5),
    ]
    process.__enter__.return_value = process
    process.__exit__.side_effect = AssertionError("Popen context exit may wait without a timeout")
    process.stdout.close.side_effect = AssertionError("A live stdout reader may hold the pipe lock")
    process.wait.side_effect = AssertionError("Unbounded process wait must not run")
    start = MagicMock(return_value=process)
    tree_kill = MagicMock(side_effect=OSError("Simulated process-tree termination failure"))
    monkeypatch.setattr(pdf_preview.subprocess, "Popen", start)
    monkeypatch.setattr(pdf_preview.subprocess, "run", tree_kill)
    # Exercise Windows tree-cleanup failure on every test host without changing
    # the global os module or running any operating-system process command.
    monkeypatch.setattr(pdf_preview, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(pdf_preview.subprocess, "CREATE_NO_WINDOW", 0, raising=False)

    with pytest.raises(DocumentError, match="预览超时"):
        pdf_preview.render_pdf_page(b"offline synthetic PDF bytes", 1)

    assert [call.kwargs["timeout"] for call in process.communicate.call_args_list] == [20, 5]
    tree_kill.assert_called_once()
    process.kill.assert_called_once()
    process.__exit__.assert_not_called()
    process.stdout.close.assert_not_called()
    process.wait.assert_not_called()


def test_preview_route_returns_controlled_422_for_render_timeout(client, monkeypatch):
    from unittest.mock import MagicMock

    from backend.app import pdf_preview
    from backend.app.errors import DocumentError

    paper = upload_pdf(client)
    message = "原文页面预览超时，请使用完整原始文件入口。"
    render = MagicMock(side_effect=DocumentError(message))
    monkeypatch.setattr(pdf_preview, "render_pdf_page", render)

    response = client.get(f"/api/papers/{paper['id']}/pages/2/preview")

    assert response.status_code == 422
    assert response.json() == {"detail": message}
    render.assert_called_once()
    assert render.call_args.args[1] == 2
