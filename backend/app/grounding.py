"""Typography-aware matching against one original evidence block.

PDF text layers often encode ligatures and word wrapping differently from a
copied quotation. Only those layout differences may be ignored here: numbers,
case, punctuation and ordinary inline hyphens must still agree with the source.
"""
from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation


# Only an ASCII hyphen between letters at one physical line break can be a
# typesetting hyphen. In particular, minus signs and numeric ranges stay intact.
_LINE_HYPHEN = re.compile(r"(?<=[A-Za-z])-[ \t]*(?:\r\n|\r|\n)[ \t]*(?=[A-Za-z])")
_WHITESPACE = re.compile(r"\s+")

# Shared by literal model grounding and the conservative rule extractor. Match a
# whole number, never the leading 1 in 1e-3 or the trailing 3 in a malformed 1,2,3.
NUMBER_PATTERN = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
NUMBER_TOKEN = re.compile(r"(?<![\w.,+\-–—])(" + NUMBER_PATTERN + r")(?!\w|,|\.\d)")
_SUPER_POWER = re.compile(r"(" + NUMBER_PATTERN + r")\s*[×·]\s*10([⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
_EXPLICIT_POWER = re.compile(r"(" + NUMBER_PATTERN + r")\s*[×·]\s*10\s*\^\s*(?:\{\s*([-+]?\d+)\s*\}|\(\s*([-+]?\d+)\s*\)|([-+]?\d+))(?!\w)")
_AMBIGUOUS_POWER = re.compile(r"(" + NUMBER_PATTERN + r")\s*[×·]\s*10\s*[-+]\s*\d+")


def numeric_text(text: str) -> str:
    """Preserve signs while normalizing their PDF typography for number checks."""
    text = _SUPER_POWER.sub(lambda m: m[1] + "e" + _typography(m[2]).replace("−", "-"), str(text))
    text = re.sub(r"(?<=[-+])\s+(?=\d|\.\d)", "", _typography(text).replace("−", "-"))
    text = _EXPLICIT_POWER.sub(lambda m: m[1] + "e" + next(g for g in m.groups()[1:] if g is not None), text)
    # Without an exponent marker, 1 × 10-3 can also be an arithmetic expression.
    # Exclude the whole ambiguous expression rather than accept its leading 1.
    return _AMBIGUOUS_POWER.sub(" UNRESOLVED_NUMBER ", text)


def decimal_number(value) -> Decimal | None:
    """Parse one finite literal without approximate equality or unit conversion."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    text = numeric_text(str(value)).strip()
    if not re.fullmatch(NUMBER_PATTERN, text):
        return None
    try:
        number = Decimal(text.replace(",", ""))
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def number_in_text(value, text: str) -> bool:
    """Whether the exact numeric value occurs as a complete source token."""
    wanted = decimal_number(value)
    return wanted is not None and any(
        decimal_number(match[1]) == wanted for match in NUMBER_TOKEN.finditer(numeric_text(text))
    )


def _typography(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\u00ad", "")


def readable_text(text: str) -> str:
    """Make PDF typography readable for model input without changing source data.

    Join only letter-to-letter word wraps and expand compatibility ligatures.
    All other line breaks and spacing remain available for paragraphs, tables
    and formulas. Callers retain the original evidence for provenance checks.
    """
    return _LINE_HYPHEN.sub("", _typography(str(text)))


def compact_text(text: str) -> str:
    """Compact literal field text, joining words split by a PDF line wrap.

    Case folding and metric aliases are the caller's separate policy. This
    function never drops ordinary inline hyphens, signs or punctuation.
    """
    return _WHITESPACE.sub("", readable_text(text))


def quote_matches_source(quote: str, source: str) -> bool:
    """Check a contiguous quotation with only source-backed layout tolerance.

    Each genuine source ``letter-\nletter`` permits a retained or omitted
    hyphen independently. This accommodates a quotation containing both
    ``state-of-the-art`` and ``careful`` without dropping inline hyphens from
    either text. The caller supplies exactly one evidence block, so this cannot
    concatenate separate blocks or skip non-whitespace source content.
    """
    if not isinstance(quote, str) or not isinstance(source, str):
        return False
    quote = _WHITESPACE.sub("", _typography(quote))
    source = _typography(source)
    if not quote:
        return False

    # A marker absent from both inputs represents only optional source hyphens;
    # an actual private-use character in a PDF cannot acquire that permission.
    marker = next((chr(code) for code in range(0xE000, 0xF900)
                   if chr(code) not in quote and chr(code) not in source), None)
    if marker is None:
        return False
    source = _WHITESPACE.sub("", _LINE_HYPHEN.sub(marker, source))
    if marker not in source:
        return quote in source
    optional = re.escape(marker) + "*"
    # A query hyphen can consume a real inline hyphen or a wrapping hyphen.
    # Other query characters can skip wrapping markers, and nothing else.
    pattern = optional.join(
        "(?:-|" + re.escape(marker) + ")" if char == "-" else re.escape(char)
        for char in quote
    )
    return re.search(pattern, source) is not None
