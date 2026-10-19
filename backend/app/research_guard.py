"""Conservative vetoes for numeric claims, never a semantic proof engine.

These checks can withhold an unsafe model judgment. They cannot turn a model
answer into a verified scientific fact, infer an unreported protocol, or convert
accuracy into error. All checks use the selected immutable quotations only.
"""
from __future__ import annotations

import re

from .grounding import NUMBER_PATTERN, decimal_number, numeric_text, readable_text


_CLAIM_NUMBER = re.compile(r"(?<![A-Za-z0-9.,+\-–—])(" + NUMBER_PATTERN + r")(?![A-Za-z0-9]|,|\.\d)")


def _has_identifier(token: str, source: str) -> bool:
    # Preserve identifier boundaries: Corpus-X is not CorpusXtra. Permit PDF
    # whitespace, hyphens and parentheses between the original components.
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", token)
    pattern = r"[\s()_\-]*".join(re.escape(part) for part in parts)
    return bool(pattern and re.search(r"(?<![A-Za-z0-9_])" + pattern + r"(?![A-Za-z0-9_]|[-_][A-Za-z0-9])", readable_text(source), re.I))


def _has_number(value: str, source: str) -> bool:
    # Compact numeric units occur in PDF text layers. Only insert matching
    # whitespace for this check; stored quotations remain byte-for-byte intact.
    source = re.sub(r"(?<=\d)(?=(?:BLEU|fps|ms|GB|MB|kHz|Hz)\b)", " ", source, flags=re.I)
    return any(decimal_number(match[1]) == decimal_number(value)
               for match in _CLAIM_NUMBER.finditer(numeric_text(source)))


def numeric_claim_gaps(statement: str, quotes: list[dict], verdict: str = "supported") -> list[str]:
    """Return reasons to abstain from a numeric support/refutation judgment.

    Scope identifiers are checked for *both* verdicts: an experiment on another
    benchmark cannot refute the requested benchmark. Numeric equality is only a
    necessary support check; it is deliberately not required for refutations.
    Paired numeric cells remain unresolved until there is a structured protocol
    binding. Mere presence of a caption is not that binding.
    """
    if verdict == "insufficient":
        return []
    statement = readable_text(statement)
    numbers = list(_CLAIM_NUMBER.finditer(numeric_text(statement)))
    if re.search(r"(?:百分之|千分之)[零〇一二两三四五六七八九十百千万点]+", statement):
        return ["命题采用中文数词表示数值，当前来源检查不能可靠解析；请使用阿拉伯数字或人工核查。"]
    if not numbers:
        return []
    source = "\n".join(str(item.get("quote", "")) for item in quotes)
    reasons = []
    # Hyphenated/versioned names and mixed-case scientific identifiers. Generic
    # prose and all-capital abbreviations are excluded. Formatting punctuation
    # and PDF ligatures may differ, but identifier components must all remain.
    for token in re.findall(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*", statement):
        if re.fullmatch(r"top[-_]?\d+", token, re.I):
            continue
        if (re.search(r"[-_\d]", token) or re.search(r"[a-z][A-Z]", token)) and not _has_identifier(token, source):
            reasons.append("所引片段缺少命题中的对象或数据集标识，不能用其他实验支持或反驳。")
            break
    conditions = [
        (r"单裁剪|single[- ]crop", r"单裁剪|single[-\s]crop"),
        (r"(?:10|十)\s*裁剪|(?:10|ten)[- ]crop", r"(?:10|十)\s*裁剪|(?:10|ten)[-\s]crop"),
        (r"验证集|validation\s+set", r"验证集|validation"),
        (r"测试集|test\s+set", r"测试集|\btest(?:ing)?\b"),
    ]
    for claim_pattern, source_pattern in conditions:
        if re.search(claim_pattern, statement, re.I) and not re.search(source_pattern, readable_text(source), re.I):
            reasons.append("所引片段缺少命题明确要求的评估划分或裁剪协议。")
            break
    # Controlled language-name aliases describe task scope, not paper-specific
    # experimental results. A result for a different translation direction is
    # neither support nor refutation for the requested direction.
    languages = {"英": "english", "德": "german", "法": "french", "中": "chinese",
                 "日": "japanese", "西": "spanish", "俄": "russian", "阿": "arabic"}
    for direction in re.finditer(r"([英德法中日西俄阿])(?:语|文)?(?:到|至|→|-)?([英德法中日西俄阿])(?:语|文)?", statement):
        first, second = direction[1], direction[2]
        english = r"\b" + languages[first] + r"[\s-]*(?:to[\s-]*)?" + languages[second] + r"\b"
        chinese = first + r"(?:语|文)?(?:到|至|→|-)?" + second + r"(?:语|文)?"
        if not re.search(english + "|" + chinese, readable_text(source), re.I):
            reasons.append("所引片段未明确对应命题的翻译语言方向，不能移接另一任务的数值。")
            break
    # Parenthesized values may encode protocols, uncertainty, or another metric.
    # Without a cell-to-protocol binding, choosing either is scientifically unsafe.
    pairs = re.finditer(r"(?<![\w-])(\d+(?:\.\d+)?)\s*%?\s*\(\s*(\d+(?:\.\d+)?)\s*%?\s*\)", readable_text(source))
    claimed_values = {decimal_number(match[1]) for match in numbers}
    if any(claimed_values & {decimal_number(pair[1]), decimal_number(pair[2])} for pair in pairs):
        reasons.append("原文含成对数值，尚未建立数值、指标列与括号内外协议的可靠对应，请查看原表人工确认。")
    directions = [
        (r"错误率|误差率|\berrors?\b|\berr\.", r"错误率|误差率|\berrors?\b|\berr\."),
        (r"准确率|正确率|\baccuracy\b|\bacc\.", r"准确率|正确率|\baccuracy\b|\bacc\."),
    ]
    for claim_pattern, source_pattern in directions:
        if re.search(claim_pattern, statement, re.I) and not re.search(source_pattern, readable_text(source), re.I):
            reasons.append("所引片段未明确提供命题所用指标方向（accuracy/error），不能仅凭同一数值判断。")
            break
    # A passage may discuss both accuracy and error for different numbers.
    # Their global co-occurrence cannot authenticate the claimed association.
    if verdict == "supported" and all(re.search(pattern, source, re.I) for _, pattern in directions):
        clauses = re.split(r"[;；。，,]|(?<!\d)\.(?!\d)|\band\b", readable_text(source), flags=re.I)
        for index, (claim_pattern, source_pattern) in enumerate(directions):
            if re.search(claim_pattern, statement, re.I) and not any(
                    re.search(source_pattern, clause, re.I)
                    and not re.search(directions[1 - index][1], clause, re.I)
                    and any(_has_number(match[1], clause) for match in numbers)
                    for clause in clauses):
                reasons.append("片段同时讨论 accuracy 与 error，尚未建立命题数值与指标方向的明确对应。")
                break
    for top in re.findall(r"top[-\s]?(\d+)", statement, re.I):
        if not re.search(r"\btop[-\s]?" + re.escape(top) + r"\b", readable_text(source), re.I):
            reasons.append("所引片段没有明确的 top-k 指标列，不能仅凭结果行判断列归属。")
            break
    if verdict == "supported" and any(not _has_number(match[1], source) for match in numbers):
        reasons.append("命题中的数值未全部出现在所引片段，不能确认完整支持。")
    variants = [r"\bbase\b|基础(?:版|模型)?", r"\bfast\b|快速(?:版|模型)?",
                r"\bbig\b|大号", r"\blarge\b|大型", r"\bsmall(?:er)?\b|小型|较小"]
    declared = [pattern for pattern in variants if re.search(pattern, statement, re.I)]
    if verdict == "supported" and len(declared) == 1:
        # A paragraph can contain several configurations and measurements. A
        # shared paragraph is not an association: the requested variant must
        # occur in a clause carrying the supported number. Multi-variant claims
        # still require model/human review; do not guess which number goes where.
        clauses = re.split(r"(?<=[.!?])\s+|(?<=[。！？])|[;；]|\b(?:while|whereas|but)\b", readable_text(source), flags=re.I)
        for match in numbers:
            measured = [clause for clause in clauses if _has_number(match[1], clause)]
            if measured and not any(re.search(declared[0], clause, re.I) for clause in measured):
                reasons.append("命题数值所在的原文结果句未明确对应所述模型配置，不能把其他版本的结果移接为支持。")
                break
    return list(dict.fromkeys(reasons))
