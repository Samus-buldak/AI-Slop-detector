"""Deterministic, loss-aware document segmentation for analysis reports.

Offsets are Python Unicode-code-point offsets into ``sources[zone]``.  They are
not offsets into the raw MIME/EML byte stream.  The source strings are included
in the result so clients can safely highlight the reported spans.

This module deliberately has no LLM/API dependency: segmentation must remain
available when the semantic endpoint is offline.
"""
from __future__ import annotations

import re
from typing import Any

from src.segmentation.segments import KINDS

SCHEMA_VERSION = "1.0"
BULLET_RE = re.compile(r"^\s*(?:[-*•‣▪]|\d+[.)]|[a-zA-Z][.)])\s+")

# Periods in these abbreviations should not create sentence boundaries.
# The list is intentionally conservative and covers common RU/EN mail text.
_ABBREVIATION_SUFFIXES = (
    "т.д.", "т.п.", "т.е.", "и т.д.", "и т.п.", "т.ч.",
    "г.т.", "ул.", "д.", "г.", "им.", "стр.", "кв.", "рис.",
    "см.", "др.", "проф.", "доц.", "акад.",
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "e.g.", "i.e.", "etc.", "vs.", "approx.",
)
_CLOSING = frozenset('"\'»”’)]}')
_SENTENCE_END = frozenset('.!?…')


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Split on blank lines while retaining exact offsets into ``text``."""
    spans: list[tuple[int, int]] = []
    start = 0
    for separator in re.finditer(r"\n[ \t]*\n+", text):
        span = _trim_span(text, start, separator.start())
        if span:
            spans.append(span)
        start = separator.end()
    span = _trim_span(text, start, len(text))
    if span:
        spans.append(span)
    return spans


def _is_abbreviation(text: str, period_index: int, token_start: int) -> bool:
    # A single-letter initial ("А. Пушкин", "J. Smith").
    token = text[token_start:period_index + 1].strip().lower()
    if re.fullmatch(r"[a-zа-яё]\.", token, flags=re.I):
        return True

    # Match the whole abbreviation suffix, allowing e.g. "... и т.д.".
    left = text[max(0, period_index - 18):period_index + 1].lower()
    return any(left.endswith(abbreviation) for abbreviation in _ABBREVIATION_SUFFIXES)


def _sentence_spans(text: str, start: int, end: int, *,
                    protected_prefix_end: int | None = None) -> list[tuple[int, int]]:
    """Return sentence-like spans without changing or discarding source text.

    ``protected_prefix_end`` excludes punctuation in a list marker such as
    ``1.`` from sentence-boundary detection.
    """
    spans: list[tuple[int, int]] = []
    cursor = start
    i = start
    while i < end:
        if text[i] not in _SENTENCE_END or (
                protected_prefix_end is not None and i < protected_prefix_end):
            i += 1
            continue

        # Consume punctuation runs ("?!", "...", etc.) and closing quotes.
        punct_end = i + 1
        while punct_end < end and text[punct_end] in _SENTENCE_END:
            punct_end += 1
        while punct_end < end and text[punct_end] in _CLOSING:
            punct_end += 1

        # A boundary must be followed by whitespace or the end of the paragraph.
        if punct_end < end and not text[punct_end].isspace():
            i = punct_end
            continue

        token_start = max(text.rfind(" ", start, i), text.rfind("\t", start, i),
                          text.rfind("\n", start, i)) + 1
        if text[i] == '.' and _is_abbreviation(text, i, token_start):
            i = punct_end
            continue

        span = _trim_span(text, cursor, punct_end)
        if span:
            spans.append(span)
        cursor = punct_end
        while cursor < end and text[cursor].isspace():
            cursor += 1
        i = cursor

    span = _trim_span(text, cursor, end)
    if span:
        spans.append(span)
    return spans


def _validate_spans(source: str, spans: list[tuple[int, int]]) -> None:
    previous_end = 0
    for start, end in spans:
        if not (0 <= start < end <= len(source)):
            raise ValueError(f"Invalid segmentation span: {(start, end)}")
        if start < previous_end:
            raise ValueError("Segmentation spans overlap or are out of order")
        previous_end = end


def _node(*, segment_id: str, parent_id: str | None, level: int,
          segment_type: str, zone: str, source: str,
          start: int, end: int, **extra: Any) -> dict[str, Any]:
    node = {
        "id": segment_id,
        "parent_id": parent_id,
        "level": level,
        "type": segment_type,
        "zone": zone,
        "start_offset": start,
        "end_offset": end,
        "text_preview": source[start:min(end, start + 240)],
        "text_truncated": end - start > 240,
    }
    node.update(extra)
    return node


def _non_whitespace_coverage(source: str, spans: list[tuple[int, int]]) -> dict[str, Any]:
    # Spans are validated as ordered and non-overlapping, so no large per-
    # character bitmap is necessary (important for multi-megabyte messages).
    total = sum(not char.isspace() for char in source)
    matched = sum(sum(not char.isspace() for char in source[start:end])
                  for start, end in spans)
    return {
        "non_whitespace_characters": total,
        "covered_characters": matched,
        "coverage": round(matched / total, 6) if total else 1.0,
        "complete": matched == total,
    }


def build_document_segmentation(
    zones: dict[str, str], *, structural_method: str = "rules",
) -> dict[str, Any]:
    """Build a stable hierarchy from already-cleaned email zones.

    The exclusive hierarchy is document zone -> paragraph -> sentence.
    Optional LLM semantic blocks are returned separately because they may
    overlap paragraph boundaries and are intended as analysis views, not as
    replacement text partitions.
    """
    sources = {kind: str(zones.get(kind) or "") for kind in KINDS}
    nodes: list[dict[str, Any]] = []
    counts = {"zones": 0, "paragraphs": 0, "sentences": 0}

    for kind in KINDS:
        source = sources[kind]
        if not source.strip():
            continue
        nodes.append(_node(
            segment_id=kind, parent_id=None, level=0,
            segment_type="structural", zone=kind, source=source,
            start=0, end=len(source),
        ))
        counts["zones"] += 1

    body = sources["body"]
    paragraph_spans = _paragraph_spans(body)
    _validate_spans(body, paragraph_spans)
    sentence_spans_all: list[tuple[int, int]] = []

    for paragraph_index, (paragraph_start, paragraph_end) in enumerate(paragraph_spans, 1):
        paragraph_text = body[paragraph_start:paragraph_end]
        lines = [line for line in paragraph_text.splitlines() if line.strip()]
        bullet_count = sum(bool(BULLET_RE.match(line)) for line in lines)
        if lines and bullet_count == len(lines):
            content_type = "list"
        elif bullet_count:
            content_type = "mixed"
        else:
            content_type = "prose"

        paragraph_id = f"body.paragraph.{paragraph_index}"
        nodes.append(_node(
            segment_id=paragraph_id, parent_id="body", level=1,
            segment_type="paragraph", zone="body", source=body,
            start=paragraph_start, end=paragraph_end,
            content_type=content_type,
        ))
        counts["paragraphs"] += 1

        sentence_items: list[tuple[int, int, str]] = []
        if content_type in {"list", "mixed"}:
            # Physical lines are safer units in lists: keep bullet markers and
            # short entries intact, and do not merge a list heading with item 1.
            line_cursor = paragraph_start
            for line in paragraph_text.splitlines(keepends=True):
                line_content_end = line_cursor + len(line.rstrip("\r\n"))
                span = _trim_span(body, line_cursor, line_content_end)
                if span:
                    marker = BULLET_RE.match(body[span[0]:span[1]])
                    line_type = "list_item" if marker else "prose"
                    protected_end = span[0] + marker.end() if marker else None
                    sentence_items.extend((a, b, line_type)
                                          for a, b in _sentence_spans(
                                              body, span[0], span[1],
                                              protected_prefix_end=protected_end))
                line_cursor += len(line)
        else:
            sentence_items = [(a, b, "prose")
                              for a, b in _sentence_spans(body, paragraph_start, paragraph_end)]

        local_spans = [(a, b) for a, b, _ in sentence_items]
        _validate_spans(body, local_spans)
        sentence_spans_all.extend(local_spans)
        for sentence_index, (sentence_start, sentence_end, sentence_type) in enumerate(sentence_items, 1):
            nodes.append(_node(
                segment_id=f"{paragraph_id}.sentence.{sentence_index}",
                parent_id=paragraph_id, level=2,
                segment_type="sentence", zone="body", source=body,
                start=sentence_start, end=sentence_end,
                content_type=sentence_type,
            ))
            counts["sentences"] += 1

    # Check that paragraph and sentence layers account for every visible
    # character. Whitespace separators intentionally remain between spans.
    paragraph_coverage = _non_whitespace_coverage(body, paragraph_spans)
    sentence_coverage = _non_whitespace_coverage(body, sentence_spans_all)

    semantic_blocks = []
    raw_blocks = zones.get("_semantic_blocks")
    if isinstance(raw_blocks, (list, tuple)):
        block_values = list(raw_blocks[:6])
    else:
        block_values = [zones.get(f"semantic_block_{index}") for index in range(1, 7)]
    search_from = 0
    for index, value in enumerate(block_values, 1):
        block = str(value or "").strip()
        if not block:
            continue
        start = body.find(block, search_from)
        if start < 0:
            # Do not emit an offset that does not actually select the text or
            # reorder repeated/overlapping model output.
            continue
        end = start + len(block)
        semantic_blocks.append(_node(
            segment_id=f"body.semantic_block.{index}", parent_id="body",
            level=1, segment_type="semantic_block", zone="body", source=body,
            start=start, end=end, index=index,
            partition=False,
        ))
        search_from = end

    return {
        "schema_version": SCHEMA_VERSION,
        "methods": {
            "structural": structural_method,
            "paragraph": "blank_line_rules",
            "sentence": "punctuation_rules",
            "semantic_blocks": "llm" if semantic_blocks else None,
        },
        "offset_unit": "unicode_codepoint",
        "offset_scope": "sources[zone]",
        "sources": sources,
        "segments": nodes,
        "semantic_blocks": semantic_blocks,
        "counts": counts,
        "coverage": {
            "paragraphs": paragraph_coverage,
            "sentences": sentence_coverage,
        },
    }
