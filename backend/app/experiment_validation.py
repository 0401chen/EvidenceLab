"""Validate persisted extraction provenance against the selected paper only."""
from __future__ import annotations

import math

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
