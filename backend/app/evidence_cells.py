"""Shared literal source spans and explicit table cells for extraction checks."""
from __future__ import annotations

import re
from .grounding import NUMBER_PATTERN


def source_fragments(evidence):
    """Bind prose sentences separately without rewriting their source bytes."""
    from .providers import _citation_fragments
    payload, fragments = [], {}
    for source_index, source in enumerate(evidence, 1):
        text = source["text"]
        if source.get("source_type") == "table":
            _, local = _citation_fragments([source])
            spans = [(item["start"], item["end"]) for item in local.values()]
        else:
            edges = [0] + [match.end() for match in re.finditer(r"[。！？!?]\s*|(?<=[.!?])\s+(?=[A-Z])", text)] + [len(text)]
            spans = []
            for begin, end in zip(edges, edges[1:]):
                if end <= begin:
                    continue
                _, local = _citation_fragments([{**source, "text": text[begin:end]}], max_chars=360)
                spans.extend((begin + item["start"], begin + item["end"]) for item in local.values())
        for begin, end in spans:
            if not text[begin:end].strip():
                continue
            handle = f"F{len(fragments) + 1}"
            fragments[handle] = {"source": source, "quote": text[begin:end], "start": begin, "end": end}
            payload.append({"id": handle, "source_group": f"S{source_index}", "paper_id": source["paper_id"],
                            "page": source.get("page"), "source_type": source.get("source_type", "text"),
                            "section": source.get("section"), "text": text[begin:end]})
    return payload, fragments



def table_cells(evidence: list[dict], fragments: dict) -> dict:
    """Expose existing explicit rectangular tables; never recover guessed cells.

    Unlike metric extraction, a column can be a benchmark. Its literal column
    label remains a separate axis and is not guessed to be an accuracy metric.
    """
    cells = {}
    for source in evidence:
        if source.get("source_type") != "table":
            continue
        lines = [line for line in source["text"].splitlines() if "|" in line]
        if len(lines) < 3:
            continue
        def parts(line):
            return [value.strip() for value in line.strip().strip("|").split("|")]
        divider = next((index for index, line in enumerate(lines)
                        if all(re.fullmatch(r":?-{2,}:?", value.replace(" ", "")) for value in parts(line))), None)
        if divider is None or divider == 0:
            continue
        header, header_line = parts(lines[divider - 1]), lines[divider - 1]
        for line in lines[divider + 1:]:
            row = parts(line)
            if len(row) != len(header) or len(row) < 2 or not row[0] or re.fullmatch(NUMBER_PATTERN, row[0]):
                continue
            for column, text in enumerate(row[1:], 1):
                # Keep the entire original cell; the contract checks the value
                # against this cell, never another number elsewhere in the row.
                time_annotation = re.fullmatch(r"\s*(" + NUMBER_PATTERN + r")\s*%?\s*\(\s*("
                                               + NUMBER_PATTERN + r")\s*(ms|s|sec)\s*\)\s*", text, re.I)
                if not time_annotation and not re.fullmatch(r"\s*" + NUMBER_PATTERN + r"\s*%?(?:\s*(?:±|\+/-)\s*" + NUMBER_PATTERN + r"\s*%?)?\s*", text):
                    continue
                row_handles = [handle for handle, binding in fragments.items()
                               if binding["source"]["id"] == source["id"] and line in binding["quote"]]
                header_handles = [handle for handle, binding in fragments.items()
                                  if binding["source"]["id"] == source["id"] and header_line in binding["quote"]]
                if not row_handles or not header_handles:
                    continue
                identifier = f"T{len(cells) + 1}"
                cells[identifier] = {
                    "cell_id": identifier, "source_id": source["id"], "paper_id": source["paper_id"],
                    "row_label": row[0], "column_label": header[column], "column_index": column,
                    "value_cell_text": text, "row_quote": line, "header_quote": header_line,
                    "column_labels": header[1:],
                    "row_handles": row_handles, "header_handles": header_handles,
                    "caption_handles": [handle for handle, binding in fragments.items()
                                        if binding["source"]["id"] in source.get("caption_evidence_ids", [])],
                }
                if time_annotation:
                    cells[identifier]["value"] = time_annotation[1]
                    cells[identifier]["time_annotation"] = {"text": text[text.index("(") + 1:text.rindex(")")],
                                                               "value": time_annotation[2], "unit": time_annotation[3]}
                if len(cells) == 160:
                    return cells
    return cells
