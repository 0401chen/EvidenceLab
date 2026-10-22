"""Validate persisted extraction provenance against the selected paper only."""
from __future__ import annotations

import math
import hashlib
import json
import re
from pathlib import Path

from .errors import ProviderError
from .grounding import quote_matches_source


def prepare_experiments(records: list[dict], paper: dict, evidence: list[dict]) -> list[dict]:
    known = {e["id"]: e for e in evidence if e.get("paper_id") == paper["id"]}
    prepared = []
    identifiers = set()
    for original in records:
        if not isinstance(original, dict):
            raise ProviderError("extraction_format", "实验结果格式无效，已有记录未修改。")
        item = dict(original)
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ProviderError("extraction_format", "实验标识无效或重复，已有记录未修改。")
        identifiers.add(identifier)
        ids = item.get("evidence_ids")
        if (item.get("paper_id") != paper["id"] or not isinstance(ids, list) or not ids
                or any(not isinstance(eid, str) or eid not in known for eid in ids)):
            raise ProviderError("grounding", "实验结果包含无法定位到当前论文的引用，已有记录未修改。")
        value = item.get("value")
        if value is not None:
            try:
                numeric_valid = not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value)
            except OverflowError:
                numeric_valid = False
            if not numeric_valid:
                raise ProviderError("extraction_format", "实验数值必须是有限数，已有记录未修改。")
        fields = item.get("field_evidence", {})
        quotes = item.get("evidence_quotes", {})
        if not isinstance(fields, dict) or not isinstance(quotes, dict):
            raise ProviderError("grounding", "实验字段证据格式无效，已有记录未修改。")
        for field_ids in fields.values():
            if not isinstance(field_ids, list) or any(not isinstance(eid, str) or eid not in ids for eid in field_ids):
                raise ProviderError("grounding", "实验字段引用必须属于该实验的原文范围，已有记录未修改。")
        for field, references in quotes.items():
            if not isinstance(references, list):
                raise ProviderError("grounding", "实验引文格式无效，已有记录未修改。")
            for ref in references:
                if (not isinstance(ref, dict) or not isinstance(ref.get("id"), str)
                        or ref["id"] not in fields.get(field, [])
                        or not isinstance(ref.get("quote"), str)
                        or not quote_matches_source(ref["quote"], known[ref["id"]]["text"])):
                    raise ProviderError("grounding", "实验字段引文与所选原文不符，已有记录未修改。")
        item["paper_title"] = paper["title"]
        item.setdefault("task", None)
        item["source_pages"] = sorted({known[eid]["page"] for eid in ids
                                       if type(known[eid].get("page")) is int and known[eid]["page"] > 0})
        item["review_required"] = bool(item.get("review_required", True))
        prepared.append(item)
    return prepared


_MEASUREMENT_FIELDS = ("method", "dataset", "metric", "value", "unit")
_PROTOCOL_FIELDS = ("task", "split", "sample_size", "conditions")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _measurement_fingerprint(item: dict) -> str:
    # Candidate IDs change on save; scientific fields and their provenance do not.
    return _digest({field: item.get(field) for field in
                    (*_MEASUREMENT_FIELDS, *_PROTOCOL_FIELDS, "paper_id", "table_context",
                     "evidence_ids", "field_evidence", "evidence_quotes", "extraction_method")})


def machine_measurement_verified(item: dict, evidence: list[dict] | None = None) -> bool:
    check = item.get("machine_verification") or {}
    return (check.get("status") == "verified" and check.get("version") == "pdf_measurement_v1"
            and check.get("record_sha256") == _measurement_fingerprint(item)
            and (evidence is None or check.get("evidence_sha256") == _digest(
                [e for e in evidence if e.get("paper_id") == item.get("paper_id")])))


def prioritize_measurements(records: list[dict]) -> list[dict]:
    """Stable priority, only from a verified literal author marker, never a score.

    No result is discarded and IDs are unchanged. A marker establishes how the
    paper labels its own row, not novelty, superiority or the identity of a row
    lacking that marker. Existing extraction limits still apply afterwards.
    """
    def priority(item):
        verified = machine_measurement_verified(item)
        own = verified and bool(re.search(r"\(\s*(?:ours|this paper|this work|this study)\s*\)",
                                          item.get("method") or "", re.I))
        return (not own, not verified)
    return sorted(records, key=priority)


def verify_pdf_measurements(records: list[dict], paper: dict, evidence: list[dict],
                            pdf_path: Path | None, expected_sha256: str | None) -> list[dict]:
    """Audit a narrow measurement scope against original PDF words, without LLMs.

    This is a second physical check, not a second independent semantic theory:
    dataset/metric scope uses the shared conservative role contract. Only fully
    bound coordinate grids are supported. Unknown protocols stay unknown and
    the legacy manual-review flag is never silently cleared. No writes occur.
    """
    import pymupdf
    from .grounding import compact_text, decimal_number
    from .analysis import _canonical_metric, _dataset_result_cell
    from .table_roles import dataset_contract, method_identity
    from .providers import _field_is_literal, _unit_matches_measurement

    own = [e for e in evidence if e.get("paper_id") == paper["id"]]
    known = {e["id"]: e for e in own}
    evidence_hash = _digest(own)
    document = None
    failure = "pdf_unavailable"
    if pdf_path and Path(pdf_path).suffix.lower() == ".pdf":
        try:
            content = Path(pdf_path).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected_sha256:
                failure = "pdf_hash_mismatch"
            else:
                document = pymupdf.open(stream=content, filetype="pdf")
        except (OSError, RuntimeError, ValueError):
            failure = "pdf_unreadable"

    def require(condition, reason):
        if not condition:
            raise ValueError(reason)

    def box_ok(box):
        return (isinstance(box, (list, tuple)) and len(box) == 4
                and all(type(v) in (float, int) and math.isfinite(v) for v in box)
                and box[0] < box[2] and box[1] < box[3])

    words_by_page = {}
    def words(page):
        require(type(page) is int and 1 <= page <= len(document), "invalid_pdf_page")
        if page not in words_by_page:
            words_by_page[page] = document[page - 1].get_text("words")
        return words_by_page[page]

    def cell_words(page, box):
        require(box_ok(box), "invalid_cell_box")
        # Metadata is rounded to .01pt. Read complete words, not clipped glyphs.
        return sorted([w for w in words(page) if w[0] >= box[0] - .02 and w[1] >= box[1] - .02
                       and w[2] <= box[2] + .02 and w[3] <= box[3] + .02], key=lambda w: (w[0], w[1]))

    checked_support = {}
    def support_matches(source):
        if source["id"] not in checked_support:
            words(source.get("page"))
            require(box_ok(source.get("bbox")), "support_without_pdf_box")
            rect = pymupdf.Rect(source["bbox"]) + (-.02, -.02, .02, .02)
            raw = document[source["page"] - 1].get_text("text", clip=rect)
            checked_support[source["id"]] = quote_matches_source(source["text"], raw)
        require(checked_support[source["id"]], "support_not_in_raw_pdf")

    result = []
    try:
        for original in records:
            item = dict(original)
            check = {"version": "pdf_measurement_v1", "status": "abstained",
                     "scope": list(_MEASUREMENT_FIELDS), "human_reviewed": False,
                     "unverified_fields": list(_PROTOCOL_FIELDS),
                     "limitations": ["仅核对论文报告的测量归属，不认证科研真实性、协议一致性或统计显著性。"],
                     "checks": []}
            item["machine_verification"] = check
            try:
                require(document is not None, failure)
                require(item.get("extraction_method") == "rule", "unsupported_extraction_path")
                require(not any(item.get(f) for f in _PROTOCOL_FIELDS), "protocol_outside_verified_scope")
                prepare_experiments([item], paper, own)
                context = item.get("table_context") or {}
                table = known.get(context.get("table_evidence_id"), {})
                require(table.get("table_structure") == "caption_bounded_dataset_grid"
                        and table.get("caption_association") == "same_page_unique_geometry", "unsupported_table")
                ids = table.get("caption_evidence_ids", [])
                require(len(ids) == 1 and ids[0] in known, "missing_caption")
                caption = known[ids[0]]
                require(caption.get("page") == table.get("page"), "cross_page_caption")
                lines = [line.strip() for line in table["text"].splitlines() if line.strip()]
                parts = lambda line: [c.strip() for c in line.strip("|").split("|")]
                headers = parts(lines[0])
                contract = dataset_contract(table, caption, headers[1:], own)
                require(contract is not None, "unproven_table_roles")
                require(_canonical_metric(contract["metric"][0]) == item.get("metric"), "metric_conflict")
                require(_unit_matches_measurement(item, own), "unit_conflict")
                for field in _MEASUREMENT_FIELDS:
                    refs = item.get("evidence_quotes", {}).get(field, [])
                    field_ids = item.get("field_evidence", {}).get(field, [])
                    require(bool(refs) and bool(field_ids) and _field_is_literal(
                        field, item.get(field), [known[eid] for eid in field_ids], refs), "field_quote_unsupported")
                # Available overlapping representations cannot contradict the grid.
                require(not any(e["id"] != table["id"] and e.get("source_type") == "table"
                                and e.get("page") == table["page"] and e.get("bbox") == table.get("bbox")
                                and compact_text(e["text"]) != compact_text(table["text"]) for e in own),
                        "conflicting_physical_table")
                column = context.get("column_index")
                require(type(column) is int and 0 < column < len(headers), "invalid_column")
                require(headers[column] == item.get("dataset") == context.get("column_label"), "dataset_column_conflict")
                require(box_ok(table.get("bbox")), "invalid_table_box")
                previous_bottom = None
                for physical_row in table.get("coordinate_rows", []):
                    positions = physical_row.get("cells", [])
                    require(len(positions) == len(headers) and all(box_ok(c.get("bbox")) for c in positions), "invalid_cell_box")
                    require(all(c["bbox"][0] >= table["bbox"][0] - .02 and c["bbox"][1] >= table["bbox"][1] - .02
                                and c["bbox"][2] <= table["bbox"][2] + .02 and c["bbox"][3] <= table["bbox"][3] + .02
                                for c in positions), "cell_outside_table")
                    row_top = min(c["bbox"][1] for c in positions)
                    require(previous_bottom is None or row_top >= previous_bottom - .02, "overlapping_physical_rows")
                    previous_bottom = max(c["bbox"][3] for c in positions)
                    require(all(a["bbox"][2] <= b["bbox"][0] + .02 for a, b in zip(positions, positions[1:])), "overlapping_physical_columns")
                matching = [(i, row) for i, row in enumerate(table.get("coordinate_rows", []))
                            if row.get("row_label") == item.get("method") == context.get("row_label")]
                require(len(matching) == 1, "method_row_conflict")
                index, row = matching[0]
                require(len(lines) == len(table["coordinate_rows"]) + 2, "grid_shape_conflict")
                cells = row["cells"]
                require(len(cells) == len(headers) and parts(lines[index + 2]) == [c["text"] for c in cells], "row_text_conflict")
                require(context.get("row_quote") == lines[index + 2] and context.get("header_quote") == lines[0], "context_quote_conflict")
                for field, literal in (("method", lines[index + 2]), ("value", lines[index + 2]), ("dataset", lines[0])):
                    require(any(ref["id"] == table["id"] and quote_matches_source(literal, ref["quote"])
                                for ref in item["evidence_quotes"][field]), "field_not_bound_to_physical_cell")
                require(any(ref["id"] == caption["id"] and quote_matches_source(caption["text"], ref["quote"])
                            for ref in item["evidence_quotes"]["metric"]), "metric_not_bound_to_caption")
                require(cells[column]["text"] == context.get("value_cell_text")
                        and cells[column]["bbox"] == context.get("value_bbox"), "value_cell_conflict")
                measurement = _dataset_result_cell(cells[column]["text"])
                require(measurement is not None and decimal_number(measurement["value"]) == decimal_number(item.get("value")), "primary_value_conflict")
                require(all(context.get(k) == measurement.get(k) for k in ("uncertainty", "time_annotation")), "annotation_conflict")
                method_ok, method_support = method_identity(item["method"], table, caption, own)
                require(method_ok, "derived_or_ambiguous_method")
                supports = [caption, *method_support, *contract["measurement_support"],
                            *[s for values in contract["dataset_support"].values() for s in values]]
                for support in supports:
                    support_matches(support)
                # Raw header lies above every data row, inside this table box.
                page = table["page"]
                require(box_ok(table.get("bbox")), "invalid_table_box")
                top = min(c["bbox"][1] for r in table["coordinate_rows"] for c in r["cells"])
                header_box = [table["bbox"][0], table["bbox"][1], table["bbox"][2], top - .03]
                header_words = cell_words(page, header_box)
                remaining = list(header_words)
                header_boxes = []
                for label in headers:
                    group = []
                    while remaining and compact_text(" ".join(w[4] for w in group)) != compact_text(label):
                        group.append(remaining.pop(0))
                    require(bool(group) and compact_text(" ".join(w[4] for w in group)) == compact_text(label), "raw_header_mismatch")
                    header_boxes.append([group[0][0], min(w[1] for w in group), group[-1][2], max(w[3] for w in group)])
                require(not remaining, "ambiguous_raw_header")
                require(max(b[1] for b in header_boxes) < min(b[3] for b in header_boxes), "header_not_one_row")
                require(max(c["bbox"][1] for c in cells) < min(c["bbox"][3] for c in cells), "cells_not_one_row")
                for j, cell in enumerate(cells):
                    require(cell.get("page") == page and cell.get("column_label") == headers[j], "cell_identity_conflict")
                    raw = cell_words(page, cell["bbox"])
                    require(raw and compact_text(" ".join(w[4] for w in raw)) == compact_text(cell["text"]), "raw_cell_mismatch")
                    if j:
                        centre = (raw[0][0] + raw[0][2]) / 2
                        centres = [(b[0] + b[2]) / 2 for b in header_boxes]
                        left = header_boxes[1][0] - 8 if j == 1 else (centres[j - 1] + centres[j]) / 2
                        right = (centres[j] + centres[j + 1]) / 2 if j + 1 < len(headers) else table["bbox"][2] + .02
                        require(left <= centre < right, "raw_column_alignment_conflict")
                check.update(status="verified", reason="raw_pdf_measurement_bound",
                             pdf_sha256=expected_sha256, evidence_sha256=evidence_hash,
                             record_sha256=_measurement_fingerprint(item),
                             checks=["literal_field_quotes", "positive_dataset_roles", "uniform_metric_and_unit",
                                     "scoped_context_conflicts", "raw_pdf_support_text", "raw_pdf_header",
                                     "raw_pdf_method_row", "raw_pdf_result_cell", "primary_value_and_annotations"],
                             source_ids=list(dict.fromkeys([table["id"], *[s["id"] for s in supports]])),
                             page=page, value_bbox=list(cells[column]["bbox"]),
                             raw_cell=cells[column]["text"])
            except (ValueError, KeyError, IndexError, TypeError, ProviderError) as exc:
                check["reason"] = str(exc) if isinstance(exc, ValueError) else "invalid_source_structure"
            result.append(item)
    finally:
        if document is not None:
            document.close()
    return result
