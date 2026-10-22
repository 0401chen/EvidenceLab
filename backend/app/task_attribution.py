"""Conservative experiment task attribution, separate from measurement review.

Only a literal author-marked row plus dataset-specific experimental statements
can establish a task. A paper's title, abstract goal, or a reused baseline cannot.
Normalized labels never stand in for the original quotations.
"""
from __future__ import annotations

import hashlib
import json
import re

from .grounding import quote_matches_source, readable_text


LABELS = {"semi-supervised node classification": "半监督节点分类",
          "node classification": "节点分类", "image classification": "图像分类",
          "object detection": "目标检测", "semantic segmentation": "语义分割"}


def _plain(text):
    return re.sub(r"\s+", " ", readable_text(text)).strip()


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def infer_task(item: dict, evidence: list[dict]) -> dict | None:
    if not re.search(r"\(\s*(?:ours|this paper|this work|this study)\s*\)", item.get("method") or "", re.I):
        return None
    context = item.get("table_context") or {}
    table = next((e for e in evidence if e["id"] == context.get("table_evidence_id")
                  and e.get("paper_id") == item.get("paper_id")), None)
    if not table or context.get("column_label") != item.get("dataset") or context.get("row_label") != item.get("method"):
        return None
    dataset = item.get("dataset")
    if not dataset or item.get("metric") != "accuracy":
        return None
    name = r"(?<![\w.-])" + re.escape(dataset) + r"(?![\w.-])"
    sources = [e for e in evidence if e.get("paper_id") == item["paper_id"]
               and e.get("source_type") == "text" and type(e.get("page")) is int
               and not re.search(r"references|bibliography|参考文献", e.get("section", ""), re.I)]
    # Scope comes from a physical heading above this table, not the parser's
    # inherited section string (which can incorrectly remain ABSTRACT).
    box = table.get("bbox")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    scopes = []
    for source in sources:
        b = source.get("bbox")
        text = _plain(source["text"])
        if (source.get("page") != table.get("page") or not b or len(b) != 4
                or not 0 <= box[1] - b[1] <= 240 or b[0] >= box[2] or b[2] <= box[0]):
            continue
        # An unsupported intervening task heading must block inheritance from
        # an earlier supported one, rather than disappearing from scope search.
        heading = re.fullmatch(r"(?:\d+(?:\.\d+)*\s+)?((?:[\w-]+\s+){0,6}(?:classification|regression|detection|segmentation|prediction|clustering))", text, re.I)
        axis = re.fullmatch(r"(Transductive|Inductive)\s+Method\s+.+", text, re.I)
        if heading or axis:
            scopes.append((b[1], source, (heading or axis)[1].lower()))
    if not scopes:
        return None
    _, scope_source, scope = max(scopes, key=lambda x: x[0])
    definitions = [(e, _plain(e["text"])) for e in sources if re.search(name, _plain(e["text"]), re.I)]
    choices = []
    for source, text in definitions:
        # Use only experimental declarations. Abstract ambitions, related work,
        # denials and hypothetical protocols cannot establish experiment scope.
        if (not re.search(r"\bwe\s+(?:utilize|use|consider|evaluate|test|train)\b", text, re.I)
                or re.search(r"\b(?:not|never|future|hypothetical|instead|unlike)\b", text, re.I)):
            continue
        if scope in {"transductive", "inductive"} and not re.match(scope + r" learning\b", text, re.I):
            continue
        tasks = {m[0].lower() for m in re.finditer(
            r"\b(?:semi-supervised node classification|node classification|image classification|object detection|semantic segmentation)\b", text, re.I)}
        if "semi-supervised node classification" in tasks:
            tasks.discard("node classification")
        # A limited labeled training subset plus explicitly labeled graph nodes
        # establishes semi-supervised node classification, not merely transduction.
        limited = re.search(r"\bonly\s+(?:use\s+)?\d+\s+(?:nodes|labels|labeled nodes)\s+per class\b", text, re.I)
        training = re.search(r"\b(?:training|train)\b", text, re.I)
        labels = re.search(r"\b(?:each (?:node|document) has a class label|nodes? .{0,30}class labels?)\b", text, re.I)
        mappings = [e for e, t in definitions if re.search(
            r"\bnodes\s+(?:are|correspond to)\s+documents\b", t, re.I)]
        subset_scope = scope == "semi-supervised node classification" or bool(
            re.search(r"\ball\b.{0,45}\bfeature", text, re.I))
        if limited and training and labels and mappings and subset_scope:
            tasks.discard("node classification")
            tasks.add("semi-supervised node classification")
            choices.append(("semi-supervised node classification", [source, mappings[0]], "dataset_labeled_node_subset"))
        for task in tasks:
            # Direct task declarations must refer to this dataset in an actual
            # experiment; multiple incompatible tasks remain unresolved.
            if task == "semi-supervised node classification" and limited and training and labels and mappings and subset_scope:
                continue
            # A task mentioned for a different experiment in the same paragraph
            # does not assign this dataset. Require an explicit grammatical link.
            from .table_roles import _names
            direct = re.search(re.escape(task) + r"\s+(?:on|using|for)\s+(?:the\s+)?([^.;]+)", text, re.I)
            headers = [c.strip() for c in table["text"].splitlines()[0].strip("|").split("|")][1:]
            if not direct or dataset not in _names(direct[1], headers, complete=False):
                continue
            choices.append((task, [source], "dataset_explicit_task"))
    if len({c[0] for c in choices}) != 1:
        return None
    task, supports, basis = choices[0]
    if scope not in {"transductive", "inductive", task}:
        return None
    supports = [*supports, scope_source]
    supports = list({s["id"]: s for s in supports}.values())
    return {"task": task, "label_zh": LABELS[task], "basis": basis,
            "quotes": [{"id": s["id"], "page": s["page"], "quote": s["text"]} for s in supports]}


def _binding(item):
    return {key: item.get(key) for key in ("paper_id", "method", "dataset", "task", "task_label_zh",
            "table_context", "evidence_ids", "field_evidence", "evidence_quotes")}


def attach_task(item: dict, result: dict, evidence: list[dict]) -> None:
    item["task"] = result["task"]
    item["task_label_zh"] = result["label_zh"]
    refs = result["quotes"]
    item["field_evidence"] = {**item.get("field_evidence", {}), "task": [r["id"] for r in refs]}
    item["evidence_quotes"] = {**item.get("evidence_quotes", {}), "task": refs}
    item["evidence_ids"] = list(dict.fromkeys([*item["evidence_ids"], *[r["id"] for r in refs]]))
    known = {e["id"]: e for e in evidence}
    item["source_pages"] = sorted({known[eid]["page"] for eid in item["evidence_ids"]})
    item["task_verification"] = {"version": "dataset_task_v1", "status": "verified", "scope": ["task"],
        "basis": result["basis"], "human_reviewed": False, "record_sha256": _hash(_binding(item)),
        "evidence_sha256": _hash([e for e in evidence if e.get("paper_id") == item["paper_id"]]),
        "limitations": ["仅确认任务归属；不认证训练划分、完整协议或跨论文可比性。"]}


def machine_task_verified(item: dict, evidence: list[dict] | None = None) -> bool:
    certificate = item.get("task_verification") or {}
    if not (certificate.get("version") == "dataset_task_v1" and certificate.get("status") == "verified"
            and certificate.get("record_sha256") == _hash(_binding(item))):
        return False
    if evidence is None:
        return True
    own = [e for e in evidence if e.get("paper_id") == item.get("paper_id")]
    expected = infer_task(item, own)
    known = {e["id"]: e for e in own}
    return (certificate.get("evidence_sha256") == _hash(own) and expected is not None
            and expected["task"] == item.get("task") and expected["label_zh"] == item.get("task_label_zh")
            and expected["quotes"] == item.get("evidence_quotes", {}).get("task")
            and all(r["id"] in known and quote_matches_source(r["quote"], known[r["id"]]["text"])
                    for r in expected["quotes"]))
