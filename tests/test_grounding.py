"""Grounding tolerates PDF typography without tolerating changed evidence."""

import pytest

from backend.app.grounding import compact_text, quote_matches_source, readable_text


def test_original_synthetic_passage_layout_matches_a_continuous_quote():
    # Original fixture prose: no excerpt from a third-party research paper.
    # Keep a long, interior quote with wrapping hyphens, a typographic ligature,
    # an ordinary hyphenated phrase, and a number that must remain unchanged.
    source = (
        "This synthetic study describes a small laboratory\n"
        "sorting task. The operators record every trial\n"
        "before the scoring procedure begins.\n"
        "The Amber clas-\n"
        "siﬁcation system follows the same hand-written\n"
        "protocol in every trial. Its care-\n"
        "ful initialization is repeated for each run, and\n"
        "a stage-based schedule remains fixed. The system\n"
        "completes the reference task in 17 rounds while\n"
        "retaining the requested label order. These values\n"
        "are invented solely for software verification."
    )
    quote = (
        "The Amber classification system follows the same hand-written protocol "
        "in every trial. Its careful initialization is repeated for each run, and "
        "a stage-based schedule remains fixed. The system completes the reference "
        "task in 17 rounds while retaining the requested label order."
    )
    assert quote_matches_source(quote, source)
    assert not quote_matches_source(quote.replace("17 rounds", "99 rounds"), source)


@pytest.mark.parametrize("quote", [
    "A state-of-the-art model needs careful initialization.",
    "A state-of-the-art model needs care-ful initialization.",
    "A stateof-the-art model needs careful initialization.",
])
def test_wrapping_hyphens_can_be_kept_or_omitted_independently(quote):
    assert quote_matches_source(
        quote, "A state-\nof-the-art model needs care-\nful initialization.",
    )


@pytest.mark.parametrize(("quote", "source"), [
    ("Accuracy is 99 percent.", "Accuracy is 85 percent."),
    ("Accuracy is 85%.", "Accuracy is 85 percent."),
    ("The value is 1.5.", "The value is -1.5."),
    ("The value is -1.5.", "The value is −1.5."),
    ("The value is 10.", "The value is 1-\n0."),
    ("noncollaborative dialogues", "non-collaborative dialogues"),
    ("non-collaborative dialogues", "noncollaborative dialogues"),
    ("Amber classification", "Amber Classification"),
    ("accuracy ... improves", "accuracy consistently improves"),
    ("accuracy improves", "accuracy consistently improves"),
    ("careful", "care-\n\nful"),
])
def test_source_content_changes_are_rejected(quote, source):
    assert not quote_matches_source(quote, source)


def test_quote_cannot_join_separate_evidence_blocks():
    quote = "Method A reached 85% on Dataset B."
    assert not quote_matches_source(quote, "Method A reached 85%")
    assert not quote_matches_source(quote, "on Dataset B.")


def test_ligatures_soft_hyphens_and_whitespace_are_layout_only():
    assert quote_matches_source("efficient classification", "efﬁcient classi\u00adﬁcation")
    assert quote_matches_source("reached 85 percent.", "reached\t85\npercent.")
    assert compact_text("Efﬁcient classi-\nfication and non-collaborative") == (
        "Efficientclassificationandnon-collaborative"
    )
    assert compact_text("rate = −1.5; range 1-\n2") == "rate=−1.5;range1-2"


def test_literal_marker_characters_are_not_treated_as_source_hyphens():
    assert not quote_matches_source("careful", "care\ue000ful")
    assert quote_matches_source("care\ue000ful", "care\ue000ful")


def test_readable_input_joins_wrapped_words_and_retains_other_layout():
    source = "  Classiﬁ-\ncation\n\nMethod\tValue\nAlphaNet\t85%\n"
    assert readable_text(source) == "  Classification\n\nMethod\tValue\nAlphaNet\t85%\n"
    assert source == "  Classiﬁ-\ncation\n\nMethod\tValue\nAlphaNet\t85%\n"


def test_readable_input_preserves_signs_inline_hyphens_and_numeric_line_breaks():
    source = "non-collaborative\r\nrate = −1.5; range 1-\n2\ncare-\r\n  ful"
    assert readable_text(source) == (
        "non-collaborative\r\nrate = −1.5; range 1-\n2\ncareful"
    )


def test_readable_input_removes_soft_hyphens_without_joining_paragraphs():
    assert readable_text("efﬁcient clas\u00adsiﬁcation\n\nSecond paragraph.") == (
        "efficient classification\n\nSecond paragraph."
    )


@pytest.mark.parametrize("quote", ["", " \n\t", None])
def test_empty_or_non_text_quotes_do_not_match(quote):
    assert not quote_matches_source(quote, "Some source text.")
