"""Local document parsing with stable, page-addressable evidence records.

PDF coordinates use the PDF viewer's top-left origin in points. Only text-layer
PDF parsing is supported; scanned documents require an external OCR step.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from statistics import median
from typing import Any

from .errors import DocumentError


def _evidence(paper_id: str, page: int, text: str, section: str,
              bbox: list[float] | None = None, source_type: str = "text",
              ordinal: int = 0) -> dict:
    digest = hashlib.sha256(f"{paper_id}:{page}:{ordinal}:{text}".encode()).hexdigest()[:18]
    return {"id": f"ev_{digest}", "paper_id": paper_id, "page": page,
            "section": section, "text": text, "bbox": bbox, "source_type": source_type}


def _chunks(text: str, limit: int = 1800) -> list[str]:
    """Bound context size without dropping or rewriting the source text."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    while len(text) > limit:
        cut = max(text.rfind("\n", 0, limit), text.rfind(". ", 0, limit),
                  text.rfind("。", 0, limit))
        if cut < limit // 3:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 3:
            cut = limit
        elif text[cut:cut + 1] == "。":
            cut += 1
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


def _heading(text: str) -> str | None:
    clean = text.strip().lstrip("#").strip()
    if "\n" in clean or len(clean) > 150:
        return None
    if len(clean) < 100 and re.fullmatch(
        r"^(?:\d+(?:\.\d+)*[.\s]+)?(?:abstract|introduction|methods?|materials(?: and methods)?|"
        r"results?|experiments?|experimental (?:setup|results|evaluation)|evaluation|training|"
        r"implementation(?: details)?|datasets?|appendix|discussion|conclusions?|references|"
        r"实验(?:设置|结果|方法)?|摘要|引言|方法|结果|讨论|结论|参考文献)[.:：]?",
        clean, re.I,
    ):
        return clean
    if re.fullmatch(r"(?:\d+(?:\.\d+)*\.?|[A-Z]\.)\s+[^\n.!?。！？]{2,110}", clean):
        return clean
    if text.lstrip().startswith("#"):
        return clean
    return None


def _title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if 3 <= len(line) <= 220:
            return re.sub(r"^(?:title|标题)\s*[:：=]\s*", "", line, flags=re.I)
    return fallback


def _numeric_matrix_markdown(words: list) -> str | None:
    """Recover only complete, consistently aligned numeric parameter matrices.

    The native detector can merge several columns into one cell. Word positions
    may restore those columns without guessing missing cells or row membership.
    Arbitrary tables, wrapped labels and partially aligned rows are left alone.
    """
    from .grounding import decimal_number

    words = [word for word in words if len(word) >= 5 and str(word[4]).strip()]
    if len(words) < 9:
        return None
    tolerance = min(3.5, median(float(word[3]) - float(word[1]) for word in words) * 0.4)
    lines = []
    for word in sorted(words, key=lambda word: ((word[1] + word[3]) / 2, word[0])):
        centre = (word[1] + word[3]) / 2
        if not lines or abs(centre - median((w[1] + w[3]) / 2 for w in lines[-1])) > tolerance:
            lines.append([])
        lines[-1].append(word)
    lines = [sorted(line, key=lambda word: word[0]) for line in lines]
    header = lines[0]
    first_number = next((index for index, word in enumerate(header) if decimal_number(word[4]) is not None), None)
    if first_number is None or first_number == 0 or len(header) - first_number < 2 or len(lines) < 3:
        return None
    numeric_headers = header[first_number:]
    if any(decimal_number(word[4]) is None for word in numeric_headers):
        return None
    centres = [(word[0] + word[2]) / 2 for word in numeric_headers]
    spacing = min(right - left for left, right in zip(centres, centres[1:]))
    if spacing < 10:
        return None
    column_tolerance = min(5, spacing * 0.15)
    label_boundary = (max(word[2] for word in header[:first_number]) + numeric_headers[0][0]) / 2
    rows = [[" ".join(str(word[4]) for word in header[:first_number]), *[str(word[4]) for word in numeric_headers]]]
    for line in lines[1:]:
        label = [word for word in line if (word[0] + word[2]) / 2 < label_boundary]
        values = [word for word in line if (word[0] + word[2]) / 2 >= label_boundary]
        if (not label or len(values) != len(centres) or any(decimal_number(word[4]) is None for word in values)
                or any(abs((word[0] + word[2]) / 2 - centre) > column_tolerance for word, centre in zip(values, centres))):
            return None
        cells = [" ".join(str(word[4]) for word in label), *[str(word[4]) for word in values]]
        if any("|" in cell or "\n" in cell for cell in cells):
            return None
        rows.append(cells)
    return "\n".join("|" + "|".join(row) + "|" for row in [rows[0], ["---"] * len(rows[0]), *rows[1:]])


def _associate_table_captions(evidence: list[dict], page: int) -> None:
    """Link explicit Table captions only when geometry gives a unique pair."""
    tables = [item for item in evidence if item["page"] == page and item.get("source_type") == "table" and item.get("bbox")]
    captions = []
    for item in evidence:
        match = re.match(r"^\s*((?:Table|Tab\.)\s+[A-Z]?\d+(?:\.\d+)*|表\s*\d+)\s*[.:：]\s", item.get("text", ""), re.I)
        if item["page"] == page and item.get("bbox") and item.get("source_type") != "table" and match:
            captions.append((item, match[1]))
    pairs = []
    for table in tables:
        left, top, right, bottom = table["bbox"]
        for caption, label in captions:
            c_left, c_top, c_right, c_bottom = caption["bbox"]
            gap = top - c_bottom if c_bottom <= top else c_top - bottom if bottom <= c_top else None
            overlap = min(right, c_right) - max(left, c_left)
            if (gap is not None and 0 <= gap <= 60 and overlap > 0
                    and overlap / min(right - left, c_right - c_left) >= 0.6):
                pairs.append((gap, table, caption, label))
    for gap, table, caption, label in pairs:
        alternatives = [other for other in pairs if other[1] is table or other[2] is caption]
        if any(other[0] < gap + 2 for other in alternatives if other[1] is not table or other[2] is not caption):
            continue
        table["caption_evidence_ids"] = [caption["id"]]
        table["table_label"] = label
        table["caption_association"] = "same_page_unique_geometry"
        # Evidence payloads are immutable. A new association gets a new identity
        # so reparsing cannot overwrite an older table record under the same ID.
        table["id"] = "ev_" + hashlib.sha256(f"{table['id']}:{caption['id']}:caption-v1".encode()).hexdigest()[:18]


def _percentage_caption_metric(text: str) -> tuple[str, bool] | None:
    """Recognize one explicit percentage-capable metric, never a benchmark name."""
    from .grounding import readable_text
    text = readable_text(text)
    matches = list(re.finditer(r"\b(?:top[- ]?\d+\s+)?(?:classification\s+)?(?:accuracy|accuracies|error(?:\s+rates?)?|precision|recall|f1(?:[- ]scores?)?|auc|auroc|rmse|mae|mse|bleu|map)\b", text, re.I))
    def metric_name(value):
        value = re.sub(r"\bclassification\s+", "", value.casefold())
        top = re.match(r"top[- ]?(\d+)\s+", value)
        if top:
            return "top" + top[1] + "_" + metric_name(value[top.end():])
        return "accuracy" if value in {"accuracy", "accuracies"} else "error_rate" if value.startswith("error") else value
    metrics = {metric_name(match[0]) for match in matches}
    if len(metrics) != 1:
        return None
    metric = next(iter(metrics))
    if metric not in {"accuracy", "error_rate", "precision", "recall", "f1", "auc", "auroc"} and not re.fullmatch(r"top\d+_(?:accuracy|error_rate)", metric):
        return None
    percent = any(re.match(r"\s*(?:\(\s*(?:in\s+)?(?:[%％]|percent(?:age)?s?)\s*\)|in\s+percent(?:age)?s?\b)",
                          text[match.end():], re.I) for match in matches)
    return metric, percent


def _word_lines(words: list) -> list[list]:
    if not words:
        return []
    height = median(float(word[3]) - float(word[1]) for word in words)
    tolerance = min(3.5, height * .38)
    lines = []
    for word in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        centre = (word[1] + word[3]) / 2
        if not lines or abs(centre - median((w[1] + w[3]) / 2 for w in lines[-1])) > tolerance:
            lines.append([])
        lines[-1].append(word)
    return [sorted(line, key=lambda w: w[0]) for line in lines]


def _aligned_result_markdown(words: list, caption_text: str | None = None, *, with_layout: bool = False) -> str | dict | None:
    """Recover a flat result grid from PDF coordinates, never from value order.

    Ruled-table detectors often merge an entire row group into one cell. This
    fallback requires an explicit identity header, complete aligned columns,
    and one physical text line per row. Wrapped/missing/overlapping rows are
    deliberately not reconstructed. Original cell spelling is retained.
    """
    words = [word for word in words if len(word) >= 5 and str(word[4]).strip()]
    if len(words) < 6:
        return None
    height = median(float(word[3]) - float(word[1]) for word in words)
    lines = _word_lines(words)
    if len(lines) < 3:
        return None
    header_groups = [[]]
    for word in lines[0]:
        if header_groups[-1] and word[0] - header_groups[-1][-1][2] >= max(6.0, height * .75):
            header_groups.append([])
        header_groups[-1].append(word)
    headers = [" ".join(str(w[4]) for w in group) for group in header_groups]
    has_metric_header = any(re.search(r"accuracy|error|err\.|mAP|F1|AUC|precision|recall|Top\s*[- ]?\d", h, re.I) for h in headers[1:])
    caption_metric = _percentage_caption_metric(caption_text or "")
    dataset_grid = not has_metric_header and caption_metric is not None
    if (len(headers) < 2 or not re.fullmatch(r"models?|methods?|networks?|architectures?|approaches?", headers[0], re.I)
            or not has_metric_header and not dataset_grid
            or dataset_grid and (len(set(headers[1:])) != len(headers) - 1
                                 or any(not re.search(r"[A-Za-z]", header) for header in headers[1:]))):
        return None
    # Header centres anchor each result column; body labels may be much longer
    # than the word 'model', so the first split is at the metric header's left.
    starts = [group[0][0] for group in header_groups]
    centres = [(group[0][0] + group[-1][2]) / 2 for group in header_groups]
    boundaries = [starts[1] - min(8, height)] + [(a + b) / 2 for a, b in zip(centres[1:], centres[2:])]
    rows = [headers]
    layout = []
    observed = [[] for _ in headers[1:]]
    for line in lines[1:]:
        cells = [[] for _ in headers]
        for word in line:
            centre = (word[0] + word[2]) / 2
            column = sum(centre >= boundary for boundary in boundaries)
            cells[column].append(word)
        if any(not cell for cell in cells):
            return None
        values = [" ".join(str(w[4]) for w in cell) for cell in cells]
        if not re.search(r"[A-Za-z]", values[0]):
            return None
        for index, (value, cell) in enumerate(zip(values[1:], cells[1:])):
            from .grounding import NUMBER_PATTERN, numeric_text
            dataset_number = (r"(?:" + NUMBER_PATTERN + r")\s*%?(?:\s*(?:±|\+/-)\s*(?:" + NUMBER_PATTERN
                              + r")\s*%?|\s*\(\s*(?:" + NUMBER_PATTERN + r")\s*(?:s|sec|ms)\s*\))?")
            pattern = r"(?:" + dataset_number + r"|[-–—])" if dataset_grid else r"(?:[+−–-]?\d+(?:\.\d+)?(?:[eE][+−-]?\d+)?(?:[%％]|[A-Za-z]+)?[†‡*]?(?:\s+\([+−–-]?\d+(?:\.\d+)?\))?|[-–—])"
            if not re.fullmatch(pattern, numeric_text(value), re.I):
                return None
            if dataset_grid and not caption_metric[1] and value not in {"-", "–", "—"} and not re.search(r"[%％]", value):
                return None
            # An uncertainty/time annotation changes cell width, not the
            # position of its primary result. Check that primary token only.
            anchor = cell[0] if dataset_grid else None
            observed[index].append((anchor[0] + anchor[2]) / 2 if anchor else (cell[0][0] + cell[-1][2]) / 2)
        if any("|" in cell or "\n" in cell for cell in values):
            return None
        rows.append(values)
        layout.append({"row_label": values[0], "cells": [{"column_label": headers[index], "text": values[index],
                       "bbox": [round(float(v), 2) for v in (min(w[0] for w in cell), min(w[1] for w in cell),
                                                               max(w[2] for w in cell), max(w[3] for w in cell))]}
                      for index, cell in enumerate(cells)]})
    # Consistent alignment, not a header guess, establishes row membership.
    if any(max(column) - min(column) > max(9, height * 1.25) for column in observed):
        return None
    markdown = "\n".join("|" + "|".join(row) + "|" for row in [rows[0], ["---"] * len(headers), *rows[1:]])
    return {"markdown": markdown, "rows": layout, "headers": headers, "dataset_columns": dataset_grid} if with_layout else markdown


def _recover_percentage_dataset_tables(page, evidence: list[dict], page_number: int) -> None:
    """Recover a complete unruled matrix below one same-page metric caption."""
    texts = [item for item in evidence if item["page"] == page_number and item.get("bbox")
             and item.get("source_type") == "text"]
    captions = [item for item in texts if re.match(r"^\s*(?:Table|Tab\.)\s+[A-Z]?\d+(?:\.\d+)*\s*[.:：]\s", item["text"], re.I)
                and _percentage_caption_metric(item["text"])]
    all_words = [word for word in page.get_text("words") if len(word) >= 5 and str(word[4]).strip()]
    proposals = []
    for caption in captions:
        left, _, right, bottom = caption["bbox"]
        if re.search(r"\bcontinued\b|续表", caption["text"], re.I):
            continue
        scoped = [word for word in all_words if left - 4 <= word[0] and word[2] <= right + 4
                  and word[1] > bottom]
        lines = _word_lines(scoped)
        headers = [index for index, line in enumerate(lines) if min(w[1] for w in line) - bottom <= 60
                   and re.fullmatch(r"models?|methods?|networks?|architectures?|approaches?", str(line[0][4]), re.I)]
        if len(headers) != 1:
            continue
        start = headers[0]
        selected = [lines[start]]
        height = median(w[3] - w[1] for w in selected[0])
        for line in lines[start + 1:]:
            previous_centre = median((w[1] + w[3]) / 2 for w in selected[-1])
            centre = median((w[1] + w[3]) / 2 for w in line)
            if centre - previous_centre > max(18, height * 2):
                break
            if centre - median((w[1] + w[3]) / 2 for w in selected[0]) > 310:
                selected = []
                break  # Reject an overlong region rather than truncate rows.
            selected.append(line)
        if not selected:
            continue
        words = [word for line in selected for word in line]
        recovered = _aligned_result_markdown(words, caption["text"], with_layout=True)
        if not recovered or not recovered["dataset_columns"]:
            continue
        box = [round(float(v), 2) for v in (min(w[0] for w in words), min(w[1] for w in words),
                                           max(w[2] for w in words), max(w[3] for w in words))]
        if getattr(page, "rect", None) is not None and box[3] > page.rect.height - 50:
            continue  # No reliable same-page ending for a table at the page edge.
        if any(other is not caption and re.match(r"^\s*(?:Table|Tab\.)\s", other["text"], re.I)
               and bottom < other["bbox"][1] < box[3] for other in texts):
            continue
        proposals.append((caption, recovered, box))
    for caption, grid, box in proposals:
        if sum(other_caption["id"] == caption["id"] or other_box == box for other_caption, _, other_box in proposals) != 1:
            continue
        if any(item["page"] == page_number and item.get("source_type") == "table" and item["text"] == grid["markdown"] for item in evidence):
            continue
        sources = [item["id"] for item in texts if max(box[0], item["bbox"][0]) < min(box[2], item["bbox"][2])
                   and max(box[1], item["bbox"][1]) < min(box[3], item["bbox"][3])]
        item = _evidence(caption["paper_id"], page_number, grid["markdown"], caption["section"], box, "table", len(evidence))
        item.update(table_structure="caption_bounded_dataset_grid", caption_evidence_ids=[caption["id"]],
                    caption_association="same_page_unique_geometry", source_text_evidence_ids=sources,
                    coordinate_rows=grid["rows"], table_label=re.match(r"^\s*((?:Table|Tab\.)\s+[A-Z]?\d+(?:\.\d+)*)", caption["text"], re.I)[1])
        for row in item["coordinate_rows"]:
            for cell in row["cells"]:
                cell["page"] = page_number
        evidence.append(item)


def _recover_caption_bounded_tables(page, evidence: list[dict], page_number: int) -> None:
    """Separate a flat logical grid from a native box that swallowed a figure.

    A numbered caption bounds horizontal scope. An explicit identity/metric
    header bounds the top. Every physical row must satisfy the existing strict
    coordinate-alignment parser. No graph labels or cells are synthesized, and
    the original text/native table records remain untouched.
    """
    _recover_percentage_dataset_tables(page, evidence, page_number)
    text_items = [item for item in evidence if item["page"] == page_number
                  and item.get("bbox") and item.get("source_type") == "text"]
    captions = [item for item in text_items if re.match(
        r"^\s*(?:Table|Tab\.)\s+[A-Z]?\d+(?:\.\d+)*\s*[.:：]\s", item["text"], re.I)]
    proposals = []
    for caption in captions:
        left, top, right, _ = caption["bbox"]
        headers = [item for item in text_items
                   if re.match(r"^\s*(?:models?|methods?|networks?|architectures?|approaches?)\s", item["text"], re.I)
                   and re.search(r"top\s*[- ]?\d|accuracy|error|mAP|F1|AUC|precision|recall", item["text"], re.I)
                   and 0 < top - item["bbox"][3] <= 220
                   and left - 4 <= item["bbox"][0] < item["bbox"][2] <= right + 4]
        for header in headers:
            if any(other is not caption and left <= other["bbox"][0] < right
                   and header["bbox"][3] < other["bbox"][1] < top for other in captions):
                continue
            clip = (left - 2, header["bbox"][1] - 1, right + 2, top - 1)
            words = page.get_text("words", clip=clip)
            recovered = _aligned_result_markdown(words)
            if not recovered:
                continue
            box = [round(float(value), 2) for value in
                   (min(w[0] for w in words), min(w[1] for w in words),
                    max(w[2] for w in words), max(w[3] for w in words))]
            proposals.append((caption, header, recovered, box))
    for caption, header, content, box in proposals:
        if sum(other_header["id"] == header["id"] or other_caption["id"] == caption["id"]
               for other_caption, other_header, _, _ in proposals) != 1:
            continue
        existing = [item for item in evidence if item["page"] == page_number and item.get("source_type") == "table"]
        if any(item["text"] == content for item in existing):
            continue
        source_ids = [item["id"] for item in text_items if box[0] - 2 <= item["bbox"][0]
                      and item["bbox"][2] <= box[2] + 2 and box[1] - 1 <= item["bbox"][1]
                      and item["bbox"][3] <= box[3] + 1]
        label = re.match(r"^\s*((?:Table|Tab\.)\s+[A-Z]?\d+(?:\.\d+)*)", caption["text"], re.I)[1]
        item = _evidence(header["paper_id"], page_number, content, header["section"], box, "table", len(evidence))
        item.update(table_structure="caption_bounded_aligned_text", header_evidence_ids=[header["id"]],
                    header_bbox=header["bbox"], source_text_evidence_ids=source_ids,
                    caption_evidence_ids=[caption["id"]], table_label=label,
                    caption_association="same_page_unique_geometry")
        item["id"] = "ev_" + hashlib.sha256(f"{item['id']}:{caption['id']}:logical-grid-v1".encode()).hexdigest()[:18]
        evidence.append(item)


def _parse_text(path: Path, paper_id: str) -> dict:
    raw = path.read_bytes()
    warnings: list[str] = []
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("gb18030")
            warnings.append("文档使用 GB18030 编码读取，请核对特殊字符。")
        except UnicodeDecodeError as exc:
            raise DocumentError("文本编码无法识别，请保存为 UTF-8 后重新上传。") from exc
    if "\x00" in text:
        raise DocumentError("文件包含二进制内容，请上传 PDF、UTF-8 TXT 或 Markdown 文件。")
    evidence: list[dict] = []
    section = "正文"
    pages = text.split("\f")
    for page, page_text in enumerate(pages, 1):
        for block in re.split(r"\n\s*\n", page_text):
            section = _heading(block) or section
            for chunk in _chunks(block):
                evidence.append(_evidence(paper_id, page, chunk, section, ordinal=len(evidence)))
    if not evidence:
        raise DocumentError("文档没有可读取的文字内容。")
    return {"title": _title(text, path.stem), "authors": [], "year": None,
            "page_count": len(pages), "evidence": evidence, "warnings": warnings,
            "parser": "text"}


def _parse_pdf(path: Path, paper_id: str) -> dict:
    try:
        import pymupdf
    except ImportError as exc:
        raise DocumentError("缺少 PyMuPDF，请安装项目基础依赖后重试。") from exc
    warnings: list[str] = []
    evidence: list[dict] = []
    section = "正文"
    try:
        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                raise DocumentError("PDF 已加密，请先解除密码保护。")
            metadata = doc.metadata or {}
            first_text = ""
            for index, page in enumerate(doc):
                page_has_text = False
                for block in page.get_text("blocks", sort=True):
                    if len(block) > 6 and block[6] != 0:
                        continue
                    content = str(block[4]).strip()
                    if not content:
                        continue
                    page_has_text = True
                    if index == 0:
                        first_text += content + "\n"
                    section = _heading(content) or section
                    bbox = [round(float(v), 2) for v in block[:4]]
                    for chunk in _chunks(content):
                        evidence.append(_evidence(paper_id, index + 1, chunk, section,
                                                  bbox, ordinal=len(evidence)))
                # PyMuPDF's native ruled-table detector is local and optional.
                # Retain a whole table, including headers and row boundaries,
                # instead of splitting every cell into unrelated prose blocks.
                if hasattr(page, "find_tables"):
                    try:
                        tables = page.find_tables().tables
                        for table in tables:
                            content = table.to_markdown().strip()
                            recovered = _numeric_matrix_markdown(page.get_text("words", clip=table.bbox))
                            if not recovered:
                                recovered = _aligned_result_markdown(page.get_text("words", clip=table.bbox))
                            if recovered:
                                content = recovered
                            if content:
                                table_bbox = [round(float(v), 2) for v in table.bbox]
                                preceding = [e for e in evidence if e["page"] == index + 1 and e.get("bbox")
                                             and e["bbox"][1] <= table_bbox[1]]
                                table_section = preceding[-1]["section"] if preceding else section
                                evidence.append(_evidence(paper_id, index + 1, content,
                                                          table_section, table_bbox, "table", len(evidence)))
                        _associate_table_captions(evidence, index + 1)
                        _recover_caption_bounded_tables(page, evidence, index + 1)
                    except Exception:
                        warnings.append(f"第 {index + 1} 页的表格结构未能可靠读取，已保留原始文本块，请对照原文核查。")
                if not page_has_text:
                    warnings.append(f"第 {index + 1} 页没有文本层，请先使用外部 OCR 生成可检索 PDF。")
            count = len(doc)
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("PDF 读取失败，文件可能损坏或格式不受支持。") from exc
    if not evidence:
        raise DocumentError("PDF 未提取到文字；扫描件请先使用外部 OCR 生成可检索 PDF。")
    warnings.append("基础解析保留文本块与页码；复杂表格、公式及双栏顺序需要人工核对。")
    author = metadata.get("author", "").strip()
    return {"title": metadata.get("title", "").strip() or _title(first_text, path.stem),
            "authors": [a.strip() for a in re.split(r";|；", author) if a.strip()],
            # PDF creation time is not the paper's publication year.
            "year": None,
            "page_count": count, "evidence": evidence, "warnings": warnings, "parser": "pymupdf"}


def parse_document(path: Path, paper_id: str, parser: str = "pymupdf") -> dict:
    path = Path(path)
    if not path.is_file():
        raise DocumentError("上传文件不存在或已被删除。")
    if path.suffix.lower() in {".txt", ".md"}:
        return _parse_text(path, paper_id)
    if path.suffix.lower() != ".pdf":
        raise DocumentError("目前支持 PDF、TXT 和 Markdown 文件。")
    if parser != "pymupdf":
        raise DocumentError("解析器无效，当前版本仅支持 pymupdf。")
    return _parse_pdf(path, paper_id)
