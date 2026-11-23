"""Interpretable, bounded contextual features (not authorship proof)."""
from __future__ import annotations

import math
import re

from src.segmentation.segments import FORWARD_HEADER, KINDS, is_quote_line, is_reply_header

CONTEXT_NAMES = (
    'log_document_words', 'log_body_words', 'log_subject_words',
    'mean_token_length', 'lexical_diversity', 'average_sentence_words',
    'question_per_sentence', 'exclamation_per_sentence', 'comma_per_word',
    'digit_fraction', 'upper_fraction', 'newline_per_word',
    'bullet_line_fraction', 'url_count', 'quote_line_fraction',
    'has_reply_marker', 'has_forward_marker', 'greeting_present',
    'signature_present', 'subject_present', 'body_present',
)
TOKEN_RE = re.compile(r"(?u)\b[\w'-]+\b")
URL_RE = re.compile(r'\b(?:https?://|www\.)[^\s<>()]+', re.I)
SENTENCE_END = re.compile(r'[.!?。！？]+(?:["»”’\x27)\]]+)?(?:\s+|$)')
BULLET_RE = re.compile(r'^\s*(?:[-*•]|\d+[.)])\s+')


def _clamp(value):
    return max(0.0, min(1.0, float(value)))


def context_features(zones: dict[str, str], original_body: str = '') -> dict[str, float]:
    """Measure author style on zones, and thread markers on the source body."""
    text = '\n'.join(zones[k] for k in KINDS if zones[k].strip())
    tokens = TOKEN_RE.findall(text)
    body_tokens = TOKEN_RE.findall(zones['body'])
    subject_tokens = TOKEN_RE.findall(zones['subject'])
    words = max(1, len(tokens))
    sentences = max(1, sum(bool(TOKEN_RE.search(part)) for part in SENTENCE_END.split(text)))
    letters = [c for c in text if c.isalpha()]
    body_lines = [line for line in original_body.splitlines() if line.strip()]
    owned_lines = [line for kind in ('greeting', 'body', 'signature')
                   for line in zones[kind].splitlines() if line.strip()]
    quote_count = sum(is_quote_line(line) for line in body_lines)
    result = (
        _clamp(math.log1p(len(tokens)) / 9),
        _clamp(math.log1p(len(body_tokens)) / 9),
        _clamp(math.log1p(len(subject_tokens)) / 6),
        _clamp(sum(map(len, tokens)) / words / 12),
        _clamp(len(set(t.casefold() for t in tokens)) / words),
        _clamp(len(tokens) / sentences / 50),
        _clamp(text.count('?') / sentences / 3),
        _clamp(text.count('!') / sentences / 3),
        _clamp(text.count(',') / words / .2),
        _clamp(sum(c.isdigit() for c in text) / max(1, len(text)) / .3),
        _clamp(sum(c.isupper() for c in letters) / max(1, len(letters)) / .8),
        _clamp(text.count('\n') / words / .3),
        _clamp(sum(bool(BULLET_RE.match(line)) for line in owned_lines) / max(1, len(owned_lines))),
        _clamp(len(URL_RE.findall(text)) / 5),
        _clamp(quote_count / max(1, len(body_lines))),
        float(bool(quote_count or any(is_reply_header(body_lines, i) for i in range(len(body_lines))))),
        float(any(FORWARD_HEADER.match(line.strip()) for line in body_lines)),
        float(bool(zones['greeting'].strip())), float(bool(zones['signature'].strip())),
        float(bool(zones['subject'].strip())), float(bool(zones['body'].strip())),
    )
    return dict(zip(CONTEXT_NAMES, result))
