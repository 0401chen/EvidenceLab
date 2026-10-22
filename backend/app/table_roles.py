"""Narrow positive contracts for recovered benchmark-column result grids.

Unknown caption language abstains. Coordinate alignment is not a semantic role.
All returned supports remain separate, literal immutable source records.
"""
from __future__ import annotations

import re
import math

from .grounding import readable_text
from .parsing import _percentage_caption_metric


def plain(text: str) -> str:
    return re.sub(r"\s+", " ", readable_text(text)).strip()


def _names(text: str, headers: list[str], *, complete: bool = True) -> list[str]:
    found = []
    def consume(match):
        header = next((h for h in headers if plain(h).casefold() == match[0].casefold()), None)
        if header is None:
            return match[0]  # Unicode regex equivalence is not identifier equality.
        found.append(header)
        return " "
    alternatives = "|".join(re.escape(plain(h)) for h in sorted(headers, key=len, reverse=True))
    remainder = re.sub(r"(?<![\w.-])(?:" + alternatives + r")(?![\w.-])", consume, text, flags=re.I)
    remainder = re.sub(r"\b(?:the|and)\b", "", remainder, flags=re.I)
    if (not found or not re.fullmatch(r"[\s,&]*", remainder) or len(found) != len(set(found))
            or complete and set(found) != set(headers)):
        return []
    return found


def _table_refs(text: str) -> set[str]:
    return {re.sub(r"\s+", "", ref.casefold()).replace("tab.", "table")
            for ref in re.findall(r"\b(?:Table|Tab\.)\s+[A-Z]?\d+(?:\.\d+)*\b", text, re.I)}


def _body_source(entry: dict, paper_id: str) -> bool:
    return (entry.get("paper_id") == paper_id and entry.get("source_type") == "text"
            and not re.search(r"references|bibliography|参考文献", entry.get("section", ""), re.I))


def table_contexts(source: dict, caption: dict, evidence: list[dict]) -> list[dict]:
    """Use explicit same-page table references, not arbitrary nearby prose.

    Chunks of one physical text block have the same bbox. Keep their immutable
    sources separate while checking the complete constraint, including a suffix
    which the PDF chunker separated from the table reference.
    """
    label = _table_refs(caption["text"])
    other_captions = {eid for item in evidence if item.get("source_type") == "table"
                      for eid in item.get("caption_evidence_ids", [])} - {caption["id"]}
    candidates = [e for e in evidence if _body_source(e, source["paper_id"])
                  and e.get("page") == source.get("page") and e["id"] != caption["id"]
                  and e["id"] not in other_captions]
    linked = [e for e in candidates if label & _table_refs(e["text"])]
    boxes = [e["bbox"] for e in linked if e.get("bbox")]
    return [e for e in candidates if e in linked or e.get("bbox") in boxes and not _table_refs(e["text"])]


def _context_consistent(contexts: list[dict], caption: dict, headers: list[str]) -> bool:
    """A column-level qualification is not a new shared table-wide contract.

    Unknown metric/unit descriptions are rejected structurally, without needing
    a list of every possible scientific metric. Unrelated table references are
    excluded by table_contexts. Multi-table qualifications remain unresolved.
    """
    label = _table_refs(caption["text"])
    for entry in contexts:
        text = plain(entry["text"])
        if _table_refs(text) - label:
            return False
        # An explicit axis declaration must itself give a dataset role. An
        # evaluation scope elsewhere cannot override a hardware/configuration axis.
        declarations = list(re.finditer(r"\b(?:result\s+)?(?:columns?|headers?|column axis)\s+(?:are|is|denote|represent|correspond to)\s+([^.;]+)", text, re.I))
        for declaration in declarations:
            role = re.fullmatch(r"(?:the\s+)?(?:datasets?|benchmarks?)(?:\s*[:—–]\s*(.+))?", declaration[1], re.I)
            if not role or role[1] and not _names(role[1], headers):
                return False
        # Consume only the proven axis statement, not other sentences or an
        # exception following it in the same block.
        for declaration in reversed(declarations):
            text = text[:declaration.start()] + " " + text[declaration.end():]
        if re.search(r"\b(?:columns?|headers?|metrics?|units?|scale|percentage|percent|accuracy|error)\b", text, re.I):
            if re.search(r"\b(?:only|except|instead|rather than|different|not|neither)\b", text, re.I):
                return False
        # Explicit descriptions of one named column are outside the supported
        # uniform table contract, even if their metric abbreviation is unknown.
        for header in headers:
            name = r"(?<![\w.-])" + re.escape(plain(header)) + r"(?![\w-]|\.[\w-])"
            if re.search(name, text, re.I):
                # Unconsumed column-specific prose is not a shared declaration.
                # Do not depend on guessing every verb or metric abbreviation.
                return False
        # A table-level declaration of unsupported measurements is not ignored.
        direct = re.search(r"\b(?:Table|Tab\.)\s+[A-Z]?\d+(?:\.\d+)*\s+(?:reports?|measures?|shows?)\s+(.+)", text, re.I)
        if direct:
            shared = (r"(?:the\s+)?(?:mean\s+)?(?:classification\s+)?(?:accuracy|accuracies|error(?:\s+rates?)?|"
                      r"precision|recall|f1|auc|auroc)(?:\s*(?:\(\s*(?:in\s+)?(?:%|percent)\s*\)|in percent))?\.?"
                      r"|(?:the\s+)?results?\.?")
            if not re.fullmatch(shared, direct[1], re.I):
                return False
    return True


def _measurement_support(metric: tuple, caption: dict, contexts: list[dict]) -> list[dict]:
    """Explicit whole-table units may be stated in its referenced body paragraph.

    Require a closed declaration about the reported numbers, not a percentage
    somewhere nearby. Context scope/conflicts have already been checked.
    """
    support = [caption]
    declaration = (r"(?:classification\s+)?(?:accuracy|accuracies|error(?:\s+rates?)?|precision|recall|f1|auc|auroc)"
                   r"\s*(?:\(\s*(?:in\s+)?(?:%|percent)\s*\)|in percent)")
    for entry in contexts:
        for statement in re.finditer(r"\b(?:reported\s+)?(?:numbers|values|results|entries)\s+"
                                     r"(?:denote|represent|are reported as|are expressed as)\s+([^.;]+)", plain(entry["text"]), re.I):
            info = _percentage_caption_metric(statement[1])
            if re.fullmatch(declaration, statement[1], re.I) and info == (metric[0], True):
                support.append(entry)
                break
    return support


def _declarations(headers: list[str], evidence: list[dict], paper_id: str, table_label: set[str]) -> dict:
    supports = {}
    for entry in evidence:
        if not _body_source(entry, paper_id) or _table_refs(entry["text"]) - table_label:
            continue
        text = plain(entry["text"])
        for header in headers:
            name = r"(?<![\w.-])" + re.escape(plain(header)) + r"(?![\w.-])"
            direct = re.search(name + r"\s+(?:dataset|benchmark)\b", text, re.I)
            described = re.search(name + r"(?:\s*\([^)]*\))?\s+is\s+(?:a|an|the)\s+"
                                  r"(?:[A-Za-z-]+\s+){0,4}(?:dataset|benchmark)\b", text, re.I)
            if direct or described:
                # Do not turn an explicit denial into a positive declaration.
                match = direct or described
                prefix = text[max(0, match.start() - 25):match.end()]
                if not re.search(r"\b(?:not|no|never|neither|unlike)\b", prefix, re.I):
                    supports.setdefault(header, entry)
        for listing in re.finditer(r"\b(?:datasets|benchmarks)\s*[:—–]\s*([^().;]+)", text, re.I):
            prefix = re.split(r"[.!?;]", text[:listing.start()])[-1]
            if re.search(r"\b(?:not|no|never|neither|unlike)\b", prefix, re.I):
                continue
            for header in _names(listing[1].strip(), headers, complete=False):
                supports.setdefault(header, entry)
    return supports


def method_identity(label: str, source: dict, caption: dict, evidence: list[dict]) -> tuple[bool, list[dict]]:
    """Distinguish a named model from a complete derived-statistic relation.

    Ordinary names use the literal Method row. A bare statistical homonym needs
    a positive model definition; a comparative/arithmetic relation cannot be
    made an absolute accuracy merely by placing it in the Method column.
    """
    text = plain(label)
    others = [plain(row["row_label"]) for row in source.get("coordinate_rows", [])
              if plain(row["row_label"]) != text]
    operands = "|".join(re.escape(value) for value in sorted(others, key=len, reverse=True))
    if operands and re.fullmatch(r"\(?\s*(?:" + operands + r")\s*[-−/+÷:]\s*(?:" + operands + r")\s*\)?", text, re.I):
        return False, []
    statistic = r"(?:gain|improvement|increase|decrease|difference|delta|diff\.?|change|ratio|mean|average|std|standard deviation|standard error)"
    stem = r"(?:(?:absolute|relative|percentage|mean|average)\s+)?" + statistic
    relation = re.fullmatch(stem + r"\s+(?:(?:over|vs\.?|versus|relative to|compared to|from|between|of|across)\s+.+"
                            r"|\([^()]*[-−/+÷:][^()]*\)|\S+\s*[-−/+÷:]\s*\S+)", text, re.I)
    if relation:
        return False, []
    ambiguous = re.fullmatch(stem + r"(?:\s*\([^()]*\))?|[∆Δ△δ]|差值|差异|比值", text, re.I)
    if not ambiguous:
        return True, []
    name = r"(?<![\w.-])" + re.escape(text) + r"(?![\w.-])"
    for entry in evidence:
        if not _body_source(entry, source["paper_id"]) or _table_refs(entry["text"]) - _table_refs(caption["text"]):
            continue
        definition = re.search(name + r"\s+is\s+(?:the name of\s+)?(?:our\s+)?(?:proposed\s+)?(?:a\s+|the\s+)?"
                               r"(?:classifier|model|method|network|architecture)\b", plain(entry["text"]), re.I)
        if definition:
            return True, [entry]
    return False, []


def same_physical_cell(left: dict, right: dict) -> bool:
    """Require the same page, caption and actual cell, with bounded PDF precision.

    Parent table boxes may have slightly different margins. Do not round them or
    merge scientific tuples globally. .02pt is the maximum accumulated error of
    coordinates stored to .01pt, and high overlap is required as well.
    """
    if any(left[key] != right[key] for key in ("paper", "page", "caption", "headers", "row", "column", "cell", "metric")):
        return False
    for key in ("caption_box", "row_box", "value_box"):
        a, b = left[key], right[key]
        if any(not math.isfinite(v) for v in (*a, *b)) or any(abs(x-y) > .020001 for x, y in zip(a, b)):
            return False
        intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
        area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
        if intersection / (area_a + area_b - intersection) < .99:
            return False
    return True


def dataset_contract(source: dict, caption: dict, headers: list[str], evidence: list[dict]) -> dict | None:
    """Require a closed whole-table metric declaration AND a positive axis role.

    Supported scopes are an explicit dataset/benchmark axis, or an exact named
    list linked to literal dataset declarations. An unscoped caption additionally
    needs a same-page paragraph explicitly referring to this table and to its
    respective datasets. No arbitrary sentence suffix is ignored.
    """
    text = plain(caption["text"])
    label = re.match(r"^((?:Table|Tab\.)\s+[A-Z]?\d+(?:\.\d+)*)\s*[:.]\s*", text, re.I)
    if not label:
        return None
    declaration, *extra = re.split(r"\.\s+(?=[A-Z])", text[label.end():], maxsplit=1)
    metric_pattern = (r"(?:top[- ]?\d+\s+)?(?:classification\s+)?(?:accurac(?:y|ies)|error(?:\s+rates?)?|"
                      r"precision|recall|f1(?:[- ]scores?)?|auc|auroc)")
    unit = r"(?:\(\s*(?:in\s+)?(?:%|percent(?:age)?s?)\s*\)|in\s+percent(?:age)?s?)"
    match = re.fullmatch(r"(?:summary\s+of\s+results\s+in\s+terms\s+of\s+)?(?P<metric>" + metric_pattern
                         + r")(?:\s*(?P<unit>" + unit + r"))?(?:\s*,?\s+(?:on|for|across)\s+(?P<scope>.+))?",
                         declaration.rstrip("."), re.I)
    if not match:
        return None
    if extra:
        # A narrowly explicit architecture footnote may define a row's feature
        # count. It cannot add a metric, unit, protocol or column qualification.
        labels = "|".join(re.escape(plain(row["row_label"])) for row in source.get("coordinate_rows", []))
        bases = {base_match[0] for row in source.get("coordinate_rows", [])
                 if (base_match := re.match(r"[\w.-]+", plain(row["row_label"])))}
        models = "|".join(re.escape(base) for base in sorted(bases, key=len, reverse=True))
        definition = (r"(?:" + labels + r")\s*(?:corresponds to|denotes)\s+the best\s+(?:" + models + r")\s+"
                      r"(?:result|model)\s+(?:computing|with)\s+\d+\s+hidden features"
                      r"(?:\s*\(using\s+[\w.-]+(?:\s+or\s+[\w.-]+)*\))?\.?")
        if not labels or not models or not re.fullmatch(definition, extra[0], re.I):
            return None
    metric = _percentage_caption_metric(match["metric"] + " " + (match["unit"] or ""))
    if not metric:
        return None
    contexts = table_contexts(source, caption, evidence)
    if not _context_consistent(contexts, caption, headers):
        return None
    def confirmed(support):
        return {"metric": metric, "dataset_support": support,
                "measurement_support": _measurement_support(metric, caption, contexts)}
    scope = match["scope"] or ""
    axis = re.fullmatch(r"(?:(?:the|these|all)\s+)?(?:(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?"
                        r"(?:benchmark\s+)?(?:datasets?|benchmarks?)(?:\s+(?P<names>.+))?", scope, re.I)
    if axis:
        if axis["count"]:
            numbers = "one two three four five six seven eight nine ten".split()
            count = int(axis["count"]) if axis["count"].isdigit() else numbers.index(axis["count"].lower()) + 1
            if count != len(headers):
                return None
        if axis["names"] and not _names(axis["names"], headers):
            return None
        if axis["names"]:
            return confirmed({h: [caption] for h in headers})
        # A count of evaluated datasets is not the identity of the column axis.
        declarations = _declarations(headers, evidence, source["paper_id"], _table_refs(caption["text"]))
        if len(declarations) != len(headers):
            return None
        return confirmed({h: [declarations[h], caption] for h in headers})
    declarations = _declarations(headers, evidence, source["paper_id"], _table_refs(caption["text"]))
    if len(declarations) != len(headers):
        return None
    if scope:
        if not _names(scope, headers):
            return None
        return confirmed({h: [declarations[h], caption] for h in headers})
    # A named declaration elsewhere does not by itself establish this table's
    # axis. Require an explicit, unambiguous reference to the current table.
    for entry in contexts:
        if (entry.get("paper_id") != source["paper_id"] or entry.get("page") != source.get("page")
                or entry.get("source_type") != "text" or entry["id"] == caption["id"]):
            continue
        context = plain(entry["text"])
        references = re.findall(r"\bTable\s+[A-Z]?\d+(?:\.\d+)*\b", context, re.I)
        if ({ref.casefold() for ref in references} == {label[1].casefold()}
                and re.search(r"\b(?:each|respective)\s+(?:dataset|benchmark)\b", context, re.I)):
            return confirmed({h: [declarations[h], entry] for h in headers})
    return None
