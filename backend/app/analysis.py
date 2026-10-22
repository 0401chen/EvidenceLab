"""Deterministic evidence retrieval, conservative extraction and alignment.

These rules describe what was reported. They do not establish scientific truth.
Every populated extraction field retains an identifier of the source block.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import unicodedata
from collections import Counter
from time import perf_counter

from .grounding import NUMBER_PATTERN, decimal_number, numeric_text, quote_matches_source, readable_text


_ALIASES = {
    "accuracy": ("accuracy", "accuracies", "accurate", "准确率", "准确度", "精度"),
    "f1": ("f1", "f1-score", "f1 score", "f1分数", "f1得分"),
    "recall": ("recall", "召回率", "敏感度"),
    "precision": ("precision", "查准率", "精确率"),
    "auc": ("auc", "auroc", "曲线下面积"),
    "dataset": ("dataset", "data set", "datasets", "数据集"),
    "method": ("method", "methods", "methodology", "方法", "模型"),
    "sample": ("sample size", "samples", "sample", "样本量", "样本数"),
    "split": ("split", "test set", "划分", "测试集", "训练集"),
    "condition": ("conditions", "condition", "实验条件", "实验设置"),
    "result": ("results", "result", "实验结果", "结果"),
    "baseline": ("baseline", "基线"),
    "rmse": ("rmse", "均方根误差"),
    "mae": ("mae", "平均绝对误差"),
    "mse": ("mse", "均方误差"),
    "error_rate": ("error rate", "error rates", "test error", "classification error", "错误率", "误差率", "测试错误率"),
    "top1_error": ("top-1 error", "top 1 error", "top1 error", "top-1错误率"),
    "top5_error": ("top-5 error", "top 5 error", "top5 error", "top-5错误率"),
    "single_model": ("single model", "single-model", "single models", "single-models", "individual model", "单模型", "单个模型", "单一模型"),
    "ensemble": ("ensemble", "ensembles", "ensembling", "集成模型", "模型集成", "集成", "组合模型"),
    "model_configuration": ("model configuration", "model configurations", "模型配置", "模型设置"),
    "ablation": ("ablation", "消融实验", "消融"),
    "training": ("training", "训练"),
    "validation": ("validation", "验证集", "验证"),
}
_STOP = set("a an and are as at be by can could did do does for from had has have how i in is it its of on or our paper papers please reported research show study studies than that the their these they this to using was were what when where which with would you your evidence compare comparison conclusion conclusions".split())
_ZH_STOP = ("请问", "请", "什么", "哪些", "多少", "如何", "是否", "为什么", "能否", "有没有", "论文", "文献", "研究", "比较", "对比", "证据", "结论", "这个", "这些", "那个", "进行", "介绍", "一下", "中的", "以及", "可以", "相关", "关于", "说明", "显示", "情况", "之间", "各自", "分别", "报告", "使用", "采用", "有没有")


def _norm(value) -> str:
    return unicodedata.normalize("NFKC", "" if value is None else str(value)).strip().casefold()


def _tokens(text: str) -> set[str]:
    text = _norm(text)
    canonical: set[str] = set()
    top_k_pattern = r"(?<![a-z0-9])top[- ]?(\d+)\s*(?:(?:test|validation|classification)\s+)?(error|accuracy|错误率|准确率)(?![a-z0-9])"
    for match in re.finditer(top_k_pattern, text):
        kind = "error" if match[2] in {"error", "错误率"} else "accuracy"
        canonical.add(f"top{match[1]}_{kind}")
    text = re.sub(top_k_pattern, " ", text)
    # Resolve the longest phrases across all concepts before their component
    # words (e.g. single model before model). This also preserves top-k errors.
    for alias, key in sorted(((alias, key) for key, aliases in _ALIASES.items()
                              for alias in aliases), key=lambda row: len(row[0]), reverse=True):
        pattern = re.escape(alias)
        if re.search(r"[a-z]", alias):
            pattern = rf"(?<![a-z0-9]){pattern}(?![a-z0-9])"
        if re.search(pattern, text):
            canonical.add(key)
            text = re.sub(pattern, " ", text)
    for phrase in _ZH_STOP:
        text = text.replace(phrase, " ")
    tokens = {t for t in re.findall(r"[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*|\d+(?:\.\d+)?", text)
              if t not in _STOP and (len(t) > 1 or t.isdigit())}
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        if len(run) > 1:
            tokens.update(run[i:i + 2] for i in range(len(run) - 1))
    return tokens | canonical


def _rank_evidence(question: str, evidence: list[dict]) -> list[tuple[float, int, dict, set[str]]]:
    query = _tokens(question)
    if not query:
        return []
    anchors = query - {"method", "result", "dataset", "condition", "sample", "split"}
    documents = [(_tokens(str(e.get("text", ""))), e) for e in evidence]
    frequency = Counter(t for tokens, _ in documents for t in tokens)
    scored = []
    for index, (tokens, item) in enumerate(documents):
        common = query & tokens
        if not common or (anchors and not anchors & common):
            continue
        # Bibliographic matches are not experimental evidence. They remain
        # searchable when the question explicitly asks about the bibliography.
        if (_is_reference(item) and not re.search(r"references?|bibliograph|参考文献|引用文献", question, re.I)):
            continue
        # IDF keeps named datasets and methods above ubiquitous metric words.
        score = sum(math.log(1 + (len(documents) + 1) / (frequency[t] + 0.5)) for t in common)
        score *= 0.65 + 0.35 * len(common) / len(query)
        score /= 1 + 0.08 * math.log1p(len(tokens))
        scored.append((score, -index, item, tokens))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return scored


def _is_reference(item: dict) -> bool:
    return bool(re.search(r"\b(?:references|bibliography)\b|参考文献", str(item.get("section", "")), re.I))


def retrieve_evidence(question: str, evidence: list[dict], limit: int = 6) -> list[dict]:
    """Lexical candidates, diversified by paper and distinct result wording.

    Neither lexical relevance nor the score establishes semantic support. A
    different number or a negated result must have a chance to reach verification
    even when many repeated supporting passages rank ahead of it.
    """
    scored = _rank_evidence(question, evidence)
    if not scored or limit <= 0:
        return []
    remaining = list(scored)
    chosen, seen_papers, covered = [], set(), set()
    while remaining and len(chosen) < min(limit, 30):
        def priority(row):
            score, order, item, tokens = row
            novelty = len(tokens - covered) / max(1, len(tokens))
            paper_bonus = 0.55 if item.get("paper_id") not in seen_papers else 0
            return (score / scored[0][0] * (0.6 + 0.4 * novelty) + paper_bonus, order)

        best = max(remaining, key=priority)
        remaining.remove(best)
        chosen.append(best[2])
        seen_papers.add(best[2].get("paper_id"))
        covered.update(best[3])
    return chosen


def select_evidence_context(question: str, evidence: list[dict], limit: int = 18,
                            max_chars: int = 30000) -> tuple[list[dict], dict]:
    """Add bounded neighbouring blocks without changing immutable quotations."""
    primary = retrieve_evidence(question, evidence, limit)
    positions = {item["id"]: i for i, item in enumerate(evidence)}
    contextual = []
    for item in primary:
        index = positions[item["id"]]
        for neighbour in evidence[max(0, index - 1):index + 2]:
            shared = (_tokens(str(neighbour.get("text", ""))) & _tokens(str(item.get("text", "")))) - {"method", "result", "dataset", "condition", "sample", "split"}
            if (neighbour.get("paper_id") == item.get("paper_id")
                    and neighbour.get("section") == item.get("section") and not _is_reference(neighbour)
                    and type(neighbour.get("page")) is int and type(item.get("page")) is int
                    and abs(neighbour["page"] - item["page"]) <= 1 and shared):
                contextual.append(neighbour)
    chosen, used, chars = [], set(), 0
    for item in primary + contextual:
        if item["id"] in used or chars + len(item.get("text", "")) > max_chars:
            continue
        chosen.append(item)
        used.add(item["id"])
        chars += len(item.get("text", ""))
    matched = _rank_evidence(question, evidence)
    diagnostics = {"matched": len(matched), "selected": len(chosen), "total": len(evidence),
                   "coverage_limited": any(row[2]["id"] not in used for row in matched),
                   "papers_covered": len({e.get("paper_id") for e in chosen}),
                   "papers_selected": len({e.get("paper_id") for e in evidence})}
    return sorted(chosen, key=lambda item: positions[item["id"]]), diagnostics


_FIELD_NAMES = {
    "experiment": "name", "experiment name": "name", "实验": "name", "实验名称": "name",
    "name": "name", "method": "method", "model": "method", "network": "method", "architecture": "method", "方法": "method", "模型": "method",
    "task": "task", "research task": "task", "研究任务": "task", "任务": "task",
    "dataset": "dataset", "data set": "dataset", "数据集": "dataset",
    "split": "split", "test split": "split", "测试集": "split", "数据划分": "split", "划分": "split",
    "metric": "metric", "指标": "metric", "评价指标": "metric",
    "value": "value", "score": "value", "result": "value", "数值": "value", "结果": "value",
    "unit": "unit", "单位": "unit", "sample size": "sample_size", "sample_size": "sample_size",
    "n": "sample_size", "样本量": "sample_size", "样本数": "sample_size",
    "conditions": "conditions", "condition": "conditions", "实验条件": "conditions", "条件": "conditions",
}
_NULLS = {"", "null", "none", "n/a", "na", "unknown", "not reported", "未报告", "未知", "无"}


def _number(value: str) -> tuple[float | None, str | None]:
    match = re.fullmatch(r"\s*(" + NUMBER_PATTERN + r")\s*(%|％)?\s*", numeric_text(value))
    if not match:
        return None, None
    exact = decimal_number(match[1])
    if exact is None:
        return None, None
    number = float(exact)
    if not math.isfinite(number) or (exact != 0 and number == 0):
        return None, None
    return number, "%" if match[2] else None


def _conditions(value: str) -> dict[str, str]:
    try:
        data = json.loads(value)
        if isinstance(data, dict):
            return {str(k).strip(): str(v).strip() for k, v in data.items()
                    if str(k).strip() and v is not None and str(v).strip()}
    except (ValueError, TypeError):
        pass
    data = {}
    for part in re.split(r"[;；]", value):
        match = re.match(r"\s*([^:=：]+?)\s*[:=：]\s*(.+?)\s*$", part)
        if match:
            data[match[1].strip()] = match[2].strip()
    if not data and _norm(value) not in _NULLS:
        data["reported"] = value.strip()
    return data


def _base_experiment(paper_id: str, source: dict, ordinal: int) -> dict:
    digest = hashlib.sha256(f"{paper_id}:{source['id']}:{ordinal}".encode()).hexdigest()[:18]
    return {"id": f"exp_{digest}", "paper_id": paper_id, "name": f"实验 {ordinal + 1}",
            "task": None, "method": None, "dataset": None, "split": None, "metric": None,
            "value": None, "unit": None, "sample_size": None, "conditions": {},
            "evidence_ids": [source["id"]], "extraction_method": "rule",
            "notes": [], "field_evidence": {}, "evidence_quotes": {}, "review_required": True,
            "source_pages": [source["page"]] if isinstance(source.get("page"), int)
            and not isinstance(source["page"], bool) and source["page"] > 0 else []}


def _record_sections(text: str) -> list[str]:
    """A second explicit experiment field must not overwrite an earlier row."""
    sections: list[str] = []
    lines: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        match = re.match(r"\s*[-*]?\s*([^:=：]{1,32}?)\s*[:=：]\s*(.*?)\s*$", line)
        key = _FIELD_NAMES.get(_norm(match[1])) if match else None
        if key and (key == "name" or key in seen) and seen:
            sections.append("\n".join(lines))
            lines, seen = [], set()
        lines.append(line)
        if key:
            seen.add(key)
    if lines:
        sections.append("\n".join(lines))
    return sections


_PROSE_CONFIGURATION_NOUN = (
    r"(?:(?:independent|random|different|separate|repeated|training|train|validation|test|"
    r"labelled|labeled|unlabelled|unlabeled|total)\s+){0,3}"
    r"(?:runs?|trials?|epochs?|iterations?|repetitions?|repeats?|seeds?|"
    r"samples?|examples?|instances?|observations?|nodes?|batches|batch(?:\s+size)?|"
    r"steps?|updates?|learning[\s-]+rate|dropout(?:[\s-]+rate)?|hidden[\s-]+units?)\b"
)
_PROSE_CONFIGURATION_SUFFIX = re.compile(
    rf"^\s*(?:(?:[-–—]\s*)?{_PROSE_CONFIGURATION_NOUN}|"
    rf"[\[(]\s*{_PROSE_CONFIGURATION_NOUN}\s*[\])])",
    re.I,
)


def _prose_measurements(pattern: str, text: str) -> list[re.Match]:
    """Keep direct metric/result pairs, not a count whose noun was truncated.

    ``accuracy of N runs`` shares the prefix of a measurement but N belongs to
    the run count. Inspect the source *after* the number (and optional percent)
    before creating a short field quote. Do not search ahead for another number
    or infer any unreported result, unit, method or dataset.
    """
    matches = []
    for match in re.finditer(pattern, text, re.I):
        end = match.end(3) if match[3] else match.end(2)
        if _PROSE_CONFIGURATION_SUFFIX.match(readable_text(text[end:])):
            continue
        matches.append(match)
    return matches


def extract_experiments(evidence: list[dict], paper_id: str) -> list[dict]:
    experiments: list[dict] = []
    physical_cells = []
    for source in evidence:
        if source.get("paper_id") != paper_id:
            continue
        if _is_reference(source):
            continue
        text = str(source.get("text", ""))
        tables = _table_experiments(source, paper_id, len(experiments))
        if not tables:
            tables = _dataset_grid_experiments(source, evidence, paper_id, len(experiments))
        if not tables:
            tables = _caption_matrix_experiments(source, evidence, paper_id, len(experiments))
        if not tables:
            tables = _paired_protocol_table_experiments(source, evidence, paper_id, len(experiments))
        if tables:
            for item in tables:
                _bind_table_caption(item, source, evidence)
                identity = item.pop("_physical_cell", None)
                if identity is not None:
                    from .table_roles import same_physical_cell
                    if any(same_physical_cell(identity, old) for old in physical_cells):
                        continue
                    physical_cells.append(identity)
                experiments.append(item)
            continue
        # Repeated fields denote a new row; never inherit settings implicitly.
        sections = _record_sections(text)
        for content in sections:
            item = _base_experiment(paper_id, source, len(experiments))
            recognized = 0
            for line in content.splitlines():
                match = re.match(r"\s*[-*]?\s*([^:=：]{1,32}?)\s*[:=：]\s*(.*?)\s*$", line)
                if not match:
                    continue
                key = _FIELD_NAMES.get(_norm(match[1]))
                value = match[2].strip()
                if key is None or _norm(value) in _NULLS:
                    continue
                recognized += 1
                if key == "value":
                    number, unit = _number(value)
                    if number is None:
                        item["notes"].append("结果数值存在区间、误差或不明确的格式，未自动转换。")
                        continue
                    item[key] = number
                    if unit:
                        if item["unit"] is not None and item["unit"] != unit:
                            item["notes"].append("原文的独立单位字段与数值后缀不一致，按数值后缀保留，需人工复核。")
                        item["unit"] = unit
                        item["field_evidence"]["unit"] = [source["id"]]
                        item["evidence_quotes"]["unit"] = [{"id": source["id"], "quote": line.strip()}]
                elif key == "unit" and item["unit"] is not None and item["unit"] != value:
                    item["notes"].append("原文的独立单位字段与数值后缀不一致，按数值后缀保留，需人工复核。")
                    continue
                elif key == "conditions":
                    item[key] = _conditions(value)
                else:
                    item[key] = value
                item["field_evidence"][key] = [source["id"]]
                item["evidence_quotes"][key] = [{"id": source["id"], "quote": line.strip()}]
            if recognized >= 2 and (item["metric"] is not None or item["value"] is not None):
                if item["metric"] is not None:
                    item["metric"] = _canonical_metric(item["metric"])
                item["notes"].append("按本证据块中的显式字段提取；未跨段合并实验条件，缺失项保留为空。")
                experiments.append(item)
                continue
            # Natural prose: only literal, unambiguous metric-number pairs.
            # No assignment of surrounding values to a named method or dataset.
            metric_pattern = r"top[- ]?\d+ (?:(?:test|validation|classification) )?(?:error|accuracy)|error rate|test error|classification error|accuracy|precision|recall|F1(?:[- ]score)?|AUROC|AUC|RMSE|MAE|MSE|错误率|误差率|准确率|精确率|召回率|均方根误差|均方误差|平均绝对误差"
            pattern = rf"(?<![a-zA-Z])({metric_pattern})\s*(?:(?:of|was|is|reached|equals?)\s+|(?:[:=：]|为|达到|达到了|是)\s*)?({NUMBER_PATTERN})(?!\w|,|\.\d)\s*(%|％)?"
            matches = _prose_measurements(pattern, numeric_text(content))
            for match_index, metric_match in enumerate(matches):
                value, _ = _number(metric_match[2])
                if value is None:
                    continue
                item = _base_experiment(paper_id, source, len(experiments))
                item.update(metric=_canonical_metric(metric_match[1]), value=value,
                            unit="%" if metric_match[3] else None)
                item["field_evidence"] = {"metric": [source["id"]], "value": [source["id"]]}
                if item["unit"]:
                    item["field_evidence"]["unit"] = [source["id"]]
                # Preserve the original spelling (including a Unicode minus)
                # rather than storing a rewritten quote from numeric_text.
                occurrences = [m for m in _prose_measurements(pattern, content)
                               if decimal_number(m[2]) == decimal_number(metric_match[2])
                               and _canonical_metric(m[1]) == item["metric"]]
                occurrence_index = sum(_canonical_metric(m[1]) == item["metric"]
                                       and decimal_number(m[2]) == decimal_number(metric_match[2])
                                       for m in matches[:match_index])
                original_match = occurrences[occurrence_index] if occurrence_index < len(occurrences) else None
                original = original_match.group(0) if original_match else content.strip()
                item["evidence_quotes"] = {f: [{"id": source["id"], "quote": original.strip()}]
                                           for f in item["field_evidence"]}
                item["notes"] = ["检测到原文指标与数值；实验归属和上下文条件未确认，需要人工补充或模型提取。"]
                item["_candidate_text"] = content
                # A single explicit configuration in the measurement's own
                # sentence can be retained literally. Do not carry it across
                # sentences or choose between single/ensemble alternatives.
                boundaries = [0] + [m.end() for m in re.finditer(r"(?<=[。！？!?])\s*|(?<=[.])\s+(?=[A-Z])", content)] + [len(content)]
                for begin, end in zip(boundaries, boundaries[1:]):
                    sentence = content[begin:end]
                    if ((original_match is not None and not begin <= original_match.start() < end)
                            or original not in sentence or len(_prose_measurements(pattern, numeric_text(sentence))) != 1):
                        continue
                    item["_candidate_text"] = sentence
                    configurations = list(re.finditer(
                        r"(?<!\w)(single[- ]model|ensemble(?:\s+of\s+\w+\s+(?:models?|networks?))?|checkpoint averaging|model averaging)(?!\w)|单模型|单个模型|集成模型|检查点平均",
                        sentence, re.I))
                    if len(configurations) == 1:
                        literal = configurations[0].group(0)
                        item["conditions"] = {literal: literal}
                        item["field_evidence"]["conditions"] = [source["id"]]
                        item["evidence_quotes"]["conditions"] = [{"id": source["id"], "quote": sentence.strip()}]
                        item["notes"].append("仅保留本测量句中唯一明确出现的模型配置原文；未推断模型名称、数据集或跨句实验设置。")
                    break
                item["_generic_candidate"] = True
                experiments.append(item)
    # Equal numbers in different source blocks do not establish an identical
    # experiment. Keep their provenance separate for review instead of dropping
    # a distinct method/split merely because its measurement happens to match.
    experiments = _suppress_weak_restatements(experiments, evidence)
    experiments.sort(key=lambda item: bool(item.get("_generic_candidate")))
    for item in experiments:
        item.pop("_generic_candidate", None)
        item.pop("_candidate_text", None)
    return experiments


def _suppress_weak_restatements(experiments: list[dict], evidence: list[dict]) -> list[dict]:
    """Avoid inventing an extra experiment from a nearby unbound restatement.

    This never merges two experiment records or transfers metadata. Only a
    metadata-free sentence naming one complete nearby record's method can be
    suppressed. Same numbers alone, another page, multiple plausible records,
    an extra known identity or a separately described protocol all retain the
    weak candidate. The immutable evidence block remains in the paper.
    """
    sources = {item["id"]: item for item in evidence}
    positions = {item["id"]: index for index, item in enumerate(evidence)}
    explicit = [item for item in experiments if not item.get("_generic_candidate")]
    identities = {str(item[field]) for item in explicit for field in ("method", "dataset", "split") if item.get(field)}
    result = []

    def literal(value: str, text: str) -> bool:
        pattern = re.escape(_norm(value))
        if value[:1].isascii() and value[:1].isalnum():
            pattern = r"(?<![a-z0-9_])" + pattern
        if value[-1:].isascii() and value[-1:].isalnum():
            pattern += r"(?![a-z0-9_])"
        return bool(re.search(pattern, _norm(text)))

    for item in experiments:
        if (not item.get("_generic_candidate") or any(item.get(field) for field in
                ("task", "method", "dataset", "split", "sample_size", "conditions"))):
            result.append(item)
            continue
        source = sources.get(item["evidence_ids"][0], {})
        text = item.get("_candidate_text", "")
        if re.search(r"\b(?:and|versus|vs|respectively|another|different|separate|independent|additional|replication|held[- ]out|cross[- ]validation)\b|分别|另一|其他|不同|独立|额外|对比", text, re.I):
            result.append(item)
            continue
        candidates = []
        for target in explicit:
            if (target.get("paper_id") != item["paper_id"] or any(not target.get(field) for field in ("method", "dataset", "split"))
                    or len(str(target["method"])) < 3 or not literal(str(target["method"]), text)
                    or any(target.get(field) != item.get(field) for field in ("metric", "value", "unit"))):
                continue
            own = [sources[eid] for eid in target["evidence_ids"] if eid in sources]
            if not any(type(source.get("page")) is int and source["page"] > 0
                       and neighbour.get("page") == source["page"]
                       and abs(positions[neighbour["id"]] - positions[source["id"]]) <= 2 for neighbour in own):
                continue
            allowed = {str(target[field]) for field in ("method", "dataset", "split")}
            if any(identity not in allowed and literal(identity, text) for identity in identities):
                continue
            # Preserve a potentially different named dataset/split even when it
            # has no explicit record elsewhere in this paper.
            named = re.findall(r"\b(?:on|using|split|partition)\s+([A-Z][A-Za-z0-9_.-]*)", text)
            named += re.findall(r"\b(?:test|validation|train(?:ing)?)[-_][A-Za-z0-9_.-]+", text, re.I)
            if any(not any(_norm(name) in _norm(value) for value in allowed) for name in named):
                continue
            candidates.append(target)
        if len(candidates) == 1:
            note = "邻接同页原文重述了该方法的相同指标与数值，但没有独立实验身份；未额外创建实验，也未将重述中的其他条件合并。原文仍保留供检索和人工核查。"
            if note not in candidates[0]["notes"]:
                candidates[0]["notes"].append(note)
        else:
            result.append(item)
    return result


def _table_experiments(source: dict, paper_id: str, offset: int) -> list[dict]:
    """Read explicit Markdown result columns; never inherit a caption's settings.

    Native PDF cell extraction supplies this representation.
    Requiring a header separator avoids interpreting arbitrary pipe-delimited
    prose or example dialogue as a result table.
    """
    lines = str(source.get("text", "")).splitlines()
    rows = []
    for header_index in range(len(lines) - 1):
        header_line, separator = lines[header_index:header_index + 2]
        if "|" not in header_line or not re.fullmatch(r"[\s|:\-]+", separator) or "---" not in separator:
            continue
        headers = [cell.strip() for cell in header_line.strip().strip("|").split("|")]
        metric_columns = []
        explicit_columns = {}
        for column, header in enumerate(headers):
            clean = re.sub(r"\s*\([%％]\)\s*|[%％]", "", header).strip()
            clean = re.sub(r"\s*\((?:test|validation|training)\)\s*$", "", clean, flags=re.I).strip()
            canonical = _canonical_metric(clean)
            if (canonical in {"accuracy", "precision", "recall", "f1", "auc", "rmse", "mae", "mse", "error_rate", "mAP"}
                    or re.fullmatch(r"top\d+_(?:error|accuracy)", canonical)
                    or re.fullmatch(r"top\s*[- ]?\d+", clean, re.I)):
                metric_columns.append((column, canonical, "%" if re.search(r"[%％]", header) else None))
            elif _norm(header) in _FIELD_NAMES:
                explicit_columns[column] = _FIELD_NAMES[_norm(header)]
        if not metric_columns:
            continue
        for row_line in lines[header_index + 2:]:
            if "|" not in row_line:
                break
            cells = [cell.strip() for cell in row_line.strip().strip("|").split("|")]
            if len(cells) != len(headers) or any(re.search(r"<br\s*/?>", cell, re.I) for cell in cells):
                continue
            for column, metric, unit in metric_columns:
                value, suffix = _number(cells[column])
                if value is None:
                    continue  # Ranges, means ± SD, footnotes and missing cells need review.
                item = _base_experiment(paper_id, source, offset + len(rows))
                item.update(metric=metric, value=value, unit=suffix or unit)
                values = {"metric": header_line, "value": row_line}
                if item["unit"]:
                    values["unit"] = row_line if suffix else header_line
                for index, field in explicit_columns.items():
                    if field in {"name", "task", "method", "dataset", "split", "sample_size"} and _norm(cells[index]) not in _NULLS:
                        item[field] = cells[index]
                        values[field] = row_line
                for field, quote in values.items():
                    item["field_evidence"][field] = [source["id"]]
                    item["evidence_quotes"][field] = [{"id": source["id"], "quote": quote.strip()}]
                if item.get("method"):
                    item["name"] = f"{item['method']} / {headers[column]}"
                split = re.search(r"\((test|validation|training)\)", headers[column], re.I)
                if split:
                    item["split"] = split[1]
                    item["field_evidence"]["split"] = [source["id"]]
                    item["evidence_quotes"]["split"] = [{"id": source["id"], "quote": header_line}]
                item["table_context"] = {"table_evidence_id": source["id"],
                                         "row_label": item.get("method"), "row_quote": row_line,
                                         "column_label": headers[column], "column_index": column,
                                         "header_quote": header_line}
                item["notes"].append("按表头和同一表格行对应提取；未把相邻正文的配置、数据集或划分自动并入。表格合并单元格、行归属和缺失条件仍需人工复核。")
                rows.append(item)
        break
    return rows


def _bind_table_caption(item: dict, source: dict, evidence: list[dict]) -> None:
    """Supply only table-wide roles explicit in a uniquely linked caption.

    Numeric values and row identity stay locked. A capitalized name alone is
    never a dataset, and a bare 'Top 1' header stays ambiguous rather than being
    silently converted to accuracy. Footnoted values need manual resolution.
    """
    ids = source.get("caption_evidence_ids", [])
    if source.get("caption_association") != "same_page_unique_geometry" or len(ids) != 1:
        return
    caption = next((e for e in evidence if e["id"] == ids[0] and e.get("paper_id") == item["paper_id"]
                    and e.get("page") == source.get("page")), None)
    if caption is None:
        return
    text = readable_text(caption["text"])
    additions = {}
    datasets = re.findall(r"\bon\s+(?:the\s+)?([A-Z][A-Za-z0-9_.-]+)\s+(?:dataset|benchmark|validation|test|training)\b", text)
    datasets += re.findall(r"\b(?:validation|test|training)\s+set\s+of\s+([A-Z][A-Za-z0-9_.-]+)\b", text)
    if len(set(datasets)) == 1:
        additions["dataset"] = datasets[0]
    splits = re.findall(r"\b(?:evaluated|tested)\s+on\s+([A-Za-z][A-Za-z0-9_-]+)", text, re.I)
    if not splits:
        splits = re.findall(r"\b(validation|test|training)\s+set\b|\b(?:[A-Z][A-Za-z0-9_.-]+)\s+(validation)\b", text)
        splits = [next(part for part in match if part) for match in splits]
        # Exceptions do not alter unmarked rows; their marked numeric cells
        # are not automatically extracted by the strict scalar parser.
        if re.search(r"\bexcept\b", text, re.I):
            splits = splits[:1]
    if len(set(splits)) == 1:
        additions["split"] = splits[0]
    tasks = re.findall(r"\b(object detection|image classification|semantic segmentation|machine translation)\s+task\b", text, re.I)
    if len(set(tasks)) == 1:
        additions["task"] = tasks[0]
    if re.search(r"\berror\s+rates?\s*\(\s*%\s*\)", text, re.I) and re.search(r"error|err\.", str(item.get("table_context", {}).get("column_label", "")), re.I):
        additions["unit"] = "%"
    configurations = re.findall(r"\b(single[- ]model|ensembles?|checkpoint averaging|model averaging)\b", text, re.I)
    if len({_norm(value) for value in configurations}) == 1 and not item.get("conditions"):
        literal = configurations[0]
        additions["conditions"] = {literal: literal}
    for field, value in additions.items():
        if item.get(field):
            continue
        item[field] = value
        item["field_evidence"][field] = [caption["id"]]
        item["evidence_quotes"][field] = [{"id": caption["id"], "quote": caption["text"]}]
        if caption["id"] not in item["evidence_ids"]:
            item["evidence_ids"].append(caption["id"])
    item.setdefault("table_context", {})["caption_evidence_ids"] = ids


def _paired_protocol_table_experiments(source: dict, evidence: list[dict], paper_id: str, offset: int) -> list[dict]:
    """Expand N (N) cells only when a unique caption explicitly maps protocols.

    Parentheses alone never establish standard deviation, a second protocol,
    percent units, or which value is primary. Shared top-k error semantics must
    also be stated in that same caption. Missing/ambiguous cells reject the
    whole paired grid instead of shifting later values into the wrong column.
    """
    ids = source.get("caption_evidence_ids", [])
    if source.get("source_type") != "table" or source.get("caption_association") != "same_page_unique_geometry" or len(ids) != 1:
        return []
    caption = next((item for item in evidence if item["id"] == ids[0]
                    and item.get("paper_id") == paper_id and item.get("page") == source.get("page")), None)
    if caption is None:
        return []
    caption_text = readable_text(caption["text"])
    protocol = r"(?:single|multi|\d+)[- ]\s*(?:crop|view)"
    mappings = list(re.finditer(r"\b(" + protocol + r")\s*\(\s*(" + protocol + r")\s*\)\s*(testing|evaluation)\b", caption_text, re.I))
    if len(mappings) != 1 or re.search(r"\b(?:accuracy|standard deviation|std|mean|confidence interval)\b", caption_text, re.I):
        return []
    if not re.search(r"\berror\s+rates?\b", caption_text, re.I):
        return []
    mapping = mappings[0]
    protocols = [re.sub(r"(?<=-)\s+", "", mapping[index]) for index in (1, 2)]
    if _norm(protocols[0]) == _norm(protocols[1]):
        return []
    lines = [line.strip() for line in source["text"].splitlines() if line.strip()]
    if len(lines) < 4 or not re.fullmatch(r"[\s|:\-]+", lines[1]):
        return []
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    if not re.fullmatch(r"models?|methods?|networks?|architectures?", headers[0], re.I):
        return []
    columns = [re.fullmatch(r"top[- ]?(\d+)", header, re.I) for header in headers[1:]]
    if not columns or any(column is None for column in columns):
        return []
    caption_top_k = set(re.findall(r"\btop[- ]?(\d+)\b", caption_text, re.I))
    if any(column[1] not in caption_top_k for column in columns):
        return []
    candidates = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(headers) or not cells[0] or any("<br" in cell.lower() for cell in cells):
            return []
        pairs = [re.fullmatch(r"\s*(" + NUMBER_PATTERN + r")\s*\(\s*(" + NUMBER_PATTERN + r")\s*\)\s*", numeric_text(cell)) for cell in cells[1:]]
        if any(pair is None for pair in pairs):
            return []
        for column_index, (column, pair) in enumerate(zip(columns, pairs), 1):
            for position, protocol_value in enumerate(protocols):
                value, _ = _number(pair[position + 1])
                if value is None:
                    return []
                item = _base_experiment(paper_id, source, offset + len(candidates))
                item.update(name=f"{cells[0]} / {headers[column_index]} / {protocol_value}", method=cells[0],
                            metric=f"top{column[1]}_error", value=value, conditions={mapping[3]: protocol_value})
                references = {"method": [(source, line)], "value": [(source, line)],
                              "metric": [(source, lines[0]), (caption, caption["text"])],
                              "conditions": [(caption, caption["text"])]}
                for field, origins in references.items():
                    item["field_evidence"][field] = [origin["id"] for origin, _ in origins]
                    item["evidence_quotes"][field] = [{"id": origin["id"], "quote": quote} for origin, quote in origins]
                item["evidence_ids"] = [source["id"], caption["id"]]
                item["table_context"] = {"table_evidence_id": source["id"], "caption_evidence_ids": ids,
                                         "header_evidence_ids": source.get("header_evidence_ids", []),
                                         "row_label": cells[0], "row_quote": line, "header_quote": lines[0],
                                         "column_label": headers[column_index], "column_index": column_index,
                                         "value_cell_text": cells[column_index], "protocol": protocol_value,
                                         "protocol_position": "outside_parentheses" if position == 0 else "inside_parentheses"}
                item["notes"].append("括号外/内数值仅依同一明确表注的两个裁剪或视图测试协议展开；未把括号猜为标准差，也未推断百分号单位。逻辑行列表和原文块均保留，仍需人工核对。")
                candidates.append(item)
    return candidates


def _dataset_result_cell(text: str) -> dict | None:
    """Read a primary result and retain explicit annotations without naming them SD.

    This is deliberately separate from _number: arbitrary prose, ranges and
    unitless parentheses do not become measurements through this table path.
    """
    scalar = r"(?P<value>" + NUMBER_PATTERN + r")\s*(?P<unit>%)?"
    match = re.fullmatch(scalar + r"(?:\s*(?P<notation>±|\+/-)\s*(?P<spread>"
                         + NUMBER_PATTERN + r")\s*(?P<spread_unit>%)?|\s*\(\s*(?P<time>"
                         + NUMBER_PATTERN + r")\s*(?P<time_unit>s|sec|ms)\s*\))?", numeric_text(text).strip(), re.I)
    if not match:
        return None
    value, _ = _number(match["value"])
    if value is None:
        return None
    result = {"value": value, "unit": "%" if match["unit"] or match["spread_unit"] else None}
    if match["spread"] is not None:
        spread, _ = _number(match["spread"])
        if spread is None or spread < 0:
            return None
        result["uncertainty"] = {"notation": match["notation"], "value": match["spread"], "kind": "unspecified"}
    if match["time"] is not None:
        time, _ = _number(match["time"])
        if time is None or time < 0:
            return None
        result["time_annotation"] = {"value": match["time"], "unit": match["time_unit"]}
    return result


def _dataset_grid_experiments(source: dict, evidence: list[dict], paper_id: str, offset: int) -> list[dict]:
    """Consume the existing coordinate-backed model-by-benchmark PDF grid.

    The parser already proved alignment/completeness. Recheck its literal layout
    against the Markdown, require a single same-page caption metric, and keep
    each result at its exact row and column. No neighbouring prose is inherited.
    """
    from .parsing import _percentage_caption_metric
    from .table_roles import dataset_contract, method_identity, plain

    ids = source.get("caption_evidence_ids", [])
    if (source.get("source_type") != "table" or source.get("table_structure") != "caption_bounded_dataset_grid"
            or source.get("caption_association") != "same_page_unique_geometry" or len(ids) != 1):
        return []
    captions = [e for e in evidence if e["id"] == ids[0] and e.get("paper_id") == paper_id
                and e.get("page") == source.get("page") and not _is_reference(e)]
    if len(captions) != 1:
        return []
    caption = captions[0]
    lines = [line.strip() for line in source["text"].splitlines() if line.strip()]
    if len(lines) < 4 or not re.fullmatch(r"[\s|:\-]+", lines[1]):
        return []
    parts = lambda line: [cell.strip() for cell in line.strip("|").split("|")]
    headers = parts(lines[0])
    configuration_headers = [re.sub(r"[_-]+", " ", header) for header in headers[1:]]
    if (len(headers) < 2 or not re.fullmatch(r"models?|methods?|networks?|architectures?|approaches?", headers[0], re.I)
            or len({_norm(header) for header in headers}) != len(headers)
            or any(not re.search(r"[A-Za-z]", header) or _percentage_caption_metric(header) for header in headers[1:])
            or any(re.search(_PROSE_CONFIGURATION_NOUN, header, re.I)
                   or re.search(r"\b(?:layers?|channels?|heads?|width|depth|hidden size|weight decay|momentum|"
                                r"regulari[sz]ation|l[12]|params?|parameters?)\b", header, re.I)
                   for header in configuration_headers)):
        return []
    contract = dataset_contract(source, caption, headers[1:], evidence)
    if contract is None:
        return []
    metric = contract["metric"]
    def physical_box(box):
        return (isinstance(box, (list, tuple)) and len(box) == 4
                and all(type(v) in (int, float) and math.isfinite(v) for v in box)
                and box[0] < box[2] and box[1] < box[3])
    if (type(source.get("page")) is not int or source["page"] < 1
            or not physical_box(source.get("bbox")) or not physical_box(caption.get("bbox"))):
        return []
    layouts = source.get("coordinate_rows", [])
    if len(layouts) != len(lines) - 2:
        return []
    parsed_rows, labels = [], set()
    for line, layout in zip(lines[2:], layouts):
        cells = parts(line)
        positions = layout.get("cells", [])
        if (len(cells) != len(headers) or len(positions) != len(headers) or not cells[0]
                or _norm(cells[0]) in labels or layout.get("row_label") != cells[0]
                or any(position.get("text") != text or position.get("column_label") != header
                       or position.get("page") != source.get("page")
                       for position, text, header in zip(positions, cells, headers))):
            return []
        labels.add(_norm(cells[0]))
        if any(not physical_box(position.get("bbox"))
               or any(position["bbox"][i] < source["bbox"][i] - .02 for i in (0, 1))
               or any(position["bbox"][i] > source["bbox"][i] + .02 for i in (2, 3)) for position in positions):
            return []
        values = []
        for cell, position in zip(cells[1:], positions[1:]):
            box = position.get("bbox", [])
            if (len(box) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
                return []
            if cell in {"-", "–", "—"}:
                values.append(None)
                continue
            value = _dataset_result_cell(cell)
            if value is None or not (value["unit"] or metric[1]):
                return []
            values.append(value)
        is_method, method_support = method_identity(cells[0], source, caption, evidence)
        if not is_method:
            continue
        parsed_rows.append((line, cells, positions, values, method_support))
    records = []
    # A dataset-major order is stable and covers model rows before moving to
    # the next benchmark. The existing API still declares any result truncation.
    for column in range(1, len(headers)):
        for line, cells, positions, values, method_support in parsed_rows:
            measurement = values[column - 1]
            if measurement is None:
                continue
            item = _base_experiment(paper_id, source, offset + len(records))
            item.update(name=f"{cells[0]} / {headers[column]}", method=cells[0], dataset=headers[column],
                        metric=_canonical_metric(metric[0]), value=measurement["value"], unit=measurement["unit"] or "%")
            provenance = {"method": (source, line), "dataset": (source, lines[0]), "value": (source, line),
                          "metric": (caption, caption["text"]),
                          "unit": (source, line) if measurement["unit"] else (caption, caption["text"])}
            for field, (origin, quote) in provenance.items():
                item["field_evidence"][field] = [origin["id"]]
                item["evidence_quotes"][field] = [{"id": origin["id"], "quote": quote}]
            item["evidence_ids"] = [source["id"], caption["id"]]
            for support in method_support:
                if support["id"] not in item["field_evidence"]["method"]:
                    item["field_evidence"]["method"].append(support["id"])
                    item["evidence_quotes"]["method"].append({"id": support["id"], "quote": support["text"]})
                if support["id"] not in item["evidence_ids"]:
                    item["evidence_ids"].append(support["id"])
            for support in contract["dataset_support"][headers[column]]:
                if support["id"] not in item["field_evidence"]["dataset"]:
                    item["field_evidence"]["dataset"].append(support["id"])
                    item["evidence_quotes"]["dataset"].append({"id": support["id"], "quote": support["text"]})
                if support["id"] not in item["evidence_ids"]:
                    item["evidence_ids"].append(support["id"])
            item["table_context"] = {"table_evidence_id": source["id"], "caption_evidence_ids": ids,
                                     "row_label": cells[0], "column_label": headers[column], "column_index": column,
                                     "row_quote": line, "header_quote": lines[0], "value_cell_text": cells[column],
                                     "value_bbox": positions[column]["bbox"]}
            item["_physical_cell"] = {"paper": paper_id, "page": source["page"], "caption": plain(caption["text"]),
                                      "headers": tuple(headers), "row": cells[0], "column": column,
                                      "cell": cells[column], "metric": metric, "caption_box": caption["bbox"],
                                      "row_box": positions[0]["bbox"], "value_box": positions[column]["bbox"]}
            for annotation in ("uncertainty", "time_annotation"):
                if annotation in measurement:
                    item["table_context"][annotation] = measurement[annotation]
            item["notes"].append("按坐标行列、明确的数据集角色依据和受限全表指标声明提取，保留完整结果单元格；±未推定为标准差、标准误或置信区间，时间未推定为训练或推理时间。是否通过机器核验及其范围请查看独立核验记录；未核验字段保持未知。")
            records.append(item)
    return records


def _caption_matrix_experiments(source: dict, evidence: list[dict], paper_id: str, offset: int) -> list[dict]:
    """Read a parameter matrix only when its unique caption names the metric.

    Row/column labels remain literal conditions; their semantic roles are not
    invented. A model/data name is populated only by an explicit local phrase.
    Each field quotes the grid or caption separately, never a fabricated join.
    """
    caption_ids = source.get("caption_evidence_ids", [])
    if (source.get("source_type") != "table" or source.get("caption_association") != "same_page_unique_geometry"
            or not isinstance(caption_ids, list) or len(caption_ids) != 1):
        return []
    caption = next((e for e in evidence if e["id"] == caption_ids[0] and e.get("paper_id") == paper_id
                    and e.get("page") == source.get("page") and not _is_reference(e)), None)
    if caption is None:
        return []
    text = readable_text(caption["text"])
    metric_pattern = r"(?:top[- ]?\d+\s+)?(?:(?:validation|test|training|classification)\s+)?(?:error(?:\s+rate)?|accuracy|precision|recall|F1(?:[- ]score)?|AUROC|AUC|RMSE|MAE|MSE)"
    measurements = list(re.finditer(r"(?<!\w)(" + metric_pattern + r")\s*\(\s*([%％])\s*\)", text, re.I))
    if len(measurements) != 1:
        return []
    measurement = measurements[0]
    metric_literal, unit = measurement[1], "%"
    # A second, explicitly different metric makes a one-metric interpretation
    # unsafe even if only the first one happens to include a unit.
    base = re.sub(r"^(?:validation|test|training|classification)\s+", "", metric_literal, flags=re.I).casefold()
    if any(re.sub(r"^(?:validation|test|training|classification)\s+", "", match.group(0), flags=re.I).casefold() != base
           for match in re.finditer(r"(?<!\w)" + metric_pattern + r"(?!\w)", text, re.I)):
        return []
    lines = [line.strip() for line in source["text"].splitlines() if line.strip()]
    if len(lines) < 4 or not re.fullmatch(r"[\s|:\-]+", lines[1]):
        return []
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    if len(headers) < 3 or not headers[0] or any(_number(cell)[0] is None for cell in headers[1:]):
        return []
    model_match = re.search(r"\b([A-Za-z][A-Za-z0-9_.-]{1,80})['’]s\s+" + re.escape(metric_literal), text, re.I)
    dataset_match = re.search(r"\b(?:dataset|benchmark|data set)\s*[:=]\s*([A-Za-z][A-Za-z0-9_.-]{1,80})\b", text, re.I)
    if dataset_match is None:
        dataset_match = re.search(r"\b(?:on|in)\s+([A-Z][A-Za-z0-9_.-]{1,80})\s+(?:dataset|benchmark)\b", text)
    split_match = re.match(r"(validation|test|training)\b", metric_literal, re.I)
    row_is_method = bool(re.search(r"\b(?:methods?|models?|architectures?|approaches?)\b|方法|模型", text, re.I))
    records = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(headers) or not cells[0]:
            return []  # A missing cell cannot be silently shifted to another column.
        row_label = cells[0]
        if (row_label in {"∆", "Δ", "△", "δ"} or re.search(r"\b(?:delta|difference|diff|ratio|gain|improvement|change)\b|差值|比值|差异|提升", row_label, re.I)):
            continue
        for column in range(1, len(cells)):
            value, suffix = _number(cells[column])
            if value is None or (suffix and suffix != unit):
                return []
            item = _base_experiment(paper_id, source, offset + len(records))
            item.update(name=f"{row_label} / {headers[0]}={headers[column]}", metric=_canonical_metric(metric_literal),
                        value=value, unit=unit, split=split_match[1].lower() if split_match else None,
                        method=row_label if row_is_method else model_match[1] if model_match else None,
                        dataset=dataset_match[1] if dataset_match else None,
                        conditions={row_label: row_label, headers[0]: headers[column]})
            provenance = {"metric": (caption, caption["text"]), "value": (source, line),
                          "unit": (caption, caption["text"]), "conditions": (source, source["text"])}
            if item["method"]:
                provenance["method"] = (source, line) if row_is_method else (caption, caption["text"])
            if item["dataset"]:
                provenance["dataset"] = (caption, caption["text"])
            if item["split"]:
                provenance["split"] = (caption, caption["text"])
            for field, (origin, quote) in provenance.items():
                item["field_evidence"][field] = [origin["id"]]
                item["evidence_quotes"][field] = [{"id": origin["id"], "quote": quote}]
            if model_match:
                item["conditions"][model_match[1]] = model_match[1]
                item["field_evidence"]["conditions"].append(caption["id"])
                item["evidence_quotes"]["conditions"].append({"id": caption["id"], "quote": caption["text"]})
            item["evidence_ids"] = [source["id"], caption["id"]]
            item["table_context"] = {"table_evidence_id": source["id"], "caption_evidence_ids": caption_ids,
                                     "row_label": row_label, "column_label": headers[column], "column_axis": headers[0]}
            item["notes"].append("依据同页唯一几何邻接的明确表号说明与完整数字矩阵对应提取。行名和列条件保留原文，未推断缺失任务、样本量或未明确命名的数据集；表注关联与行列语义仍需人工复核。")
            records.append(item)
    return records


def _canonical_metric(metric: str) -> str:
    clean = _norm(metric)
    if re.fullmatch(r"classification accurac(?:y|ies)", clean):
        return "accuracy"
    top_k = re.fullmatch(r"top[- ]?(\d+)\s+(?:(?:test|validation|classification)\s+)?(error|err\.?|accuracy)", clean)
    if top_k:
        return f"top{top_k[1]}_{'error' if top_k[2].startswith('err') else top_k[2]}"
    if clean == "map":
        return "mAP"
    for key in ("accuracy", "precision", "recall", "f1", "auc", "rmse", "mae", "mse", "error_rate", "top1_error", "top5_error"):
        if clean in _ALIASES[key]:
            return key
    return metric.strip()


_CONDITION_METADATA = {"claim", "direction", "claim_direction", "assertion", "结论", "方向"}


def _protocol(conditions: dict) -> dict:
    return {_norm(k): _norm(v) for k, v in conditions.items()
            if _norm(k) not in _CONDITION_METADATA and _norm(v) not in _NULLS}


def _claim(conditions: dict) -> tuple[str, str]:
    normalized = {_norm(k): _norm(v) for k, v in conditions.items()}
    claim = normalized.get("claim", normalized.get("assertion", normalized.get("结论", "")))
    direction = normalized.get("direction", normalized.get("claim_direction", normalized.get("方向", "")))
    positives = {"support", "supports", "positive", "支持", "成立"}
    negatives = {"oppose", "opposes", "contradict", "contradicts", "negative", "反对", "不成立"}
    return claim, "support" if direction in positives else "oppose" if direction in negatives else ""


def _same_field(field: str, left, right) -> bool:
    if field in {"value", "sample_size"}:
        first, second = decimal_number(left), decimal_number(right)
        if first is not None and second is not None:
            return first == second
    if field == "metric":
        return _norm(_canonical_metric(str(left))) == _norm(_canonical_metric(str(right)))
    return _norm(left) == _norm(right)


def _grounded_claim(item: dict, known: dict[str, dict], claim: str, direction: str) -> list[dict]:
    """Only explicit source assertions can qualify a rule-based conflict flag.

    This deliberately recognizes a narrow form of support/opposition, rather
    than inferring scientific contradiction from different numbers or labels.
    """
    matches = []
    for ref in item.get("evidence_quotes", {}).get("conditions", []):
        if not isinstance(ref, dict) or ref.get("id") not in item.get("field_evidence", {}).get("conditions", []):
            continue
        source = known.get(ref.get("id"))
        quote = ref.get("quote")
        if (not source or source.get("paper_id") != item["paper_id"] or not isinstance(quote, str)
                or not quote_matches_source(quote, source["text"])):
            continue
        text = re.sub(r"\s+", " ", _norm(quote))
        target = re.escape(re.sub(r"\s+", " ", claim))
        # Admit an explicit assertion, never arbitrary intervening clauses such
        # as "supports investigating whether ..." or "a different hypothesis".
        english_link = r"\s+(?:(?:the|a)\s+)?(?:(?:claim|hypothesis|assertion|conclusion)\s+)?(?:that\s+)?"
        chinese_link = r"(?:了)?(?:该|这一)?(?:命题|结论|假设)?[：:\s]*"
        positive = re.search(r"(?:\b(?:supports?|supported|confirms?|confirmed)\b" + english_link + r"|(?:支持|证实)" + chinese_link + ")" + target, text)
        negative = re.search(r"(?:\b(?:contradicts?|contradicted|refutes?|refuted|opposes?|opposed)\b" + english_link + r"|(?:反对|否定)" + chinese_link + ")" + target, text)
        # Absence of support, or a negated reporting verb, is not an opposite
        # conclusion. Ambiguous/negative clauses are left for human review.
        negated = re.search(r"\b(?:no|not|never|cannot|neither|without|lacks?|lacked|whether|may|might|could|uncertain|undecided|if)\b|不|未|无|可能|如果", text)
        if not negated and ((direction == "support" and positive and not negative) or (direction == "oppose" and negative)):
            matches.append({"id": ref["id"], "paper_id": item["paper_id"], "quote": quote, "direction": direction})
    return matches


def compare_experiments(experiments: list[dict], evidence: list[dict] | None = None) -> list[dict]:
    results: list[dict] = []
    known = {e["id"]: e for e in evidence or []}
    labels = {"task": "研究任务", "dataset": "数据集", "split": "数据划分", "metric": "评价指标", "unit": "单位", "method": "方法", "value": "结果值", "sample_size": "样本量"}
    # An unreported sample count prevents controlled numerical comparison, but
    # need not hide an otherwise explicit benchmark/metric alignment. Keep it
    # visible as unresolved rather than guessing the benchmark's usual size.
    critical = tuple(field for field in labels if field != "sample_size")
    for left, right in itertools.combinations(experiments, 2):
        if left.get("paper_id") == right.get("paper_id"):
            continue
        evidence_ids = list(dict.fromkeys(left.get("evidence_ids", []) + right.get("evidence_ids", [])))
        reasons: list[str] = []
        differences: list[str] = []
        missing = [labels[f] for f in critical if any(
            e.get(f) is None or _norm(e.get(f)) in _NULLS
            for e in (left, right))]
        left_conditions = left.get("conditions") or {}
        right_conditions = right.get("conditions") or {}
        left_protocol, right_protocol = _protocol(left_conditions), _protocol(right_conditions)
        if not left_protocol or not right_protocol:
            missing.append("实验条件")
        for field, label in labels.items():
            if left.get(field) is not None and right.get(field) is not None and not _same_field(field, left[field], right[field]):
                differences.append(f"{label}不同：{left[field]} / {right[field]}")
        if left_protocol != right_protocol:
            differences.append("实验条件不同：" + json.dumps(left_protocol, ensure_ascii=False) + " / " + json.dumps(right_protocol, ensure_ascii=False))
        delta = None
        left_value, right_value = decimal_number(left.get("value")), decimal_number(right.get("value"))
        numeric_valid = left_value is not None and right_value is not None
        if not numeric_valid and "结果值" not in missing:
            missing.append("有效结果值")
        incompatible = [labels[f] for f in ("task", "dataset", "split", "metric", "unit")
                        if left.get(f) is not None and right.get(f) is not None
                        and not _same_field(f, left[f], right[f])]
        if missing:
            comparability = "insufficient"
            verdict = "insufficient"
            reasons.append("缺少明确的" + "、".join(dict.fromkeys(missing)) + "，无法确认实验条件一致。")
        elif incompatible:
            comparability = "incomparable"
            verdict = "conditions_differ"
            reasons.append("、".join(incompatible) + "不一致，不直接计算差值或进行性能排名；未执行单位转换。")
        elif any(item.get("sample_size") is None or _norm(item.get("sample_size")) in _NULLS
                 for item in (left, right)):
            comparability = "conditional"
            verdict = "insufficient"
            reasons.append("已报告的任务、数据集、划分、指标和单位可以对齐，但至少一项样本量未报告；仅展示报告层面的有条件对齐，不计算差值、排名或统计显著性。")
            if _norm(left["method"]) != _norm(right["method"]) or left_protocol != right_protocol:
                reasons.append("方法或已报告配置也存在差异，不能将结果变化归因于某一个因素。")
        elif (_norm(left["method"]) != _norm(right["method"]) or left_protocol != right_protocol
              or (left.get("sample_size") is not None and right.get("sample_size") is not None
                  and not _same_field("sample_size", left["sample_size"], right["sample_size"]))):
            comparability = "conditional"
            verdict = "conditions_differ"
            reasons.append("方法、样本量或实验条件存在差异；需要确认控制变量后再解释结果差异。")
        else:
            comparability = "direct"
            delta = float(right_value - left_value)
            if not math.isfinite(delta):
                delta = None
                reasons.append("差值超出浮点显示范围，未输出近似差值。")
            verdict = "consistent" if left_value == right_value else "different"
            reasons.append("已报告的数据集、划分、指标、单位、方法与实验条件一致，可比较报告的数值。")
            claim_left, direction_left = _claim(left_conditions)
            claim_right, direction_right = _claim(right_conditions)
            if claim_left and claim_left == claim_right and {direction_left, direction_right} == {"support", "oppose"}:
                source_left = _grounded_claim(left, known, claim_left, direction_left)
                source_right = _grounded_claim(right, known, claim_right, direction_right)
                if source_left and source_right:
                    verdict = "potential_conflict"
                    reasons.append("相同可比条件下发现对同一命题的显式相反原文；仅标记潜在冲突，仍需人工核验。")
                else:
                    reasons.append("虽有相反命题标签，但缺少可定位的相反原文论断，不标记为冲突。")
            if verdict == "different":
                reasons.append("数值不同本身不构成科学结论冲突；尚未检验统计显著性与测量不确定性。")
        digest = hashlib.sha256(f"{left['id']}:{right['id']}".encode()).hexdigest()[:18]
        conflict_evidence = []
        if verdict == "potential_conflict":
            conflict_evidence = source_left + source_right
        alignment = {field: {"left": left.get(field), "right": right.get(field),
                             "status": "missing" if any(e.get(field) is None or _norm(e.get(field)) in _NULLS for e in (left, right))
                             else "same" if _same_field(field, left[field], right[field]) else "different"} for field in labels}
        results.append({"id": f"cmp_{digest}", "left_id": left["id"], "right_id": right["id"],
                        "comparability": comparability, "verdict": verdict, "reasons": reasons,
                        "differences": differences, "evidence_ids": evidence_ids, "delta": delta,
                        "alignment": alignment, "conflict_evidence": conflict_evidence, "unit_conversion": None})
    return results


def answer_local(question: str, evidence: list[dict]) -> dict:
    start = perf_counter()
    found = retrieve_evidence(question, evidence)
    if not found:
        answer = "在所选文献中未找到与问题匹配的证据。请尝试更具体的指标、数据集或方法名称。"
    else:
        paper_count = len({item["paper_id"] for item in found})
        answer = (f"检索到 {len(found)} 条相关原文，来自 {paper_count} 篇文献。"
                  "请点击下方引用查看原文与页码。当前为本地关键词检索，尚未生成语义分析或结论判断。")
    return {"answer": answer, "mode": "local", "citations": found,
            "traces": [{"agent": "证据检索代理", "status": "completed", "summary": f"检索到 {len(found)} 条相关原文；保留来源与页码。",
                        "evidence_ids": [e["id"] for e in found], "duration_ms": round((perf_counter() - start) * 1000)}],
            "warnings": ["本地模式提供原文检索，不执行语义推断。"],
            "verdict": "evidence_found" if found else "insufficient"}
