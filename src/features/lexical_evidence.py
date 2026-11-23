"""Interpretable phrase-level signals for cautious AI-authorship analysis.

The patterns below are weak stylistic indicators, not proof of authorship.
They are used as explicit model inputs so the decision tree can expose when a
phrase-density feature actually participates in its decision path.
"""
from __future__ import annotations

import re
from typing import Any

# (id, strength, regex, explanation). Phrases are intentionally multilingual.
# Common business language is marked weak; only explicit self-identification as
# a model is strong. The latter can also appear in a quotation, so it is still
# never treated as conclusive evidence.
_MARKERS = (
    ('explicit_ai_identity', 'strong', r'\bas an?\s+(?:ai\s+)?(?:language\s+)?model\b',
     'Явное самообозначение текста как ответа языковой модели'),
    ('explicit_ai_identity_ru', 'strong', r'\b(?:я\s+)?(?:являюсь\s+)?(?:языков(?:ая|ой)\s+модель|искусственный\s+интеллект)\b',
     'Явное самообозначение текста как ответа языковой модели'),
    ('no_personal_experience', 'strong', r'\b(?:i\s+do\s+not|i\s+don.t)\s+have\s+(?:personal|firsthand)\s+experiences?\b',
     'Оговорка об отсутствии личного опыта, типичная для ответов ассистента'),
    ('no_personal_experience_ru', 'strong', r'\bне\s+обладаю\s+(?:личным|собственным)\s+опытом\b',
     'Оговорка об отсутствии личного опыта, типичная для ответов ассистента'),
    ('formulaic_opening', 'weak', r'\bi\s+hope\s+this\s+(?:email|message)\s+finds\s+you\s+well\b',
     'Шаблонное деловое приветствие; часто используется и людьми'),
    ('formulaic_opening_ru', 'weak', r'\bнадеюсь,?\s+это\s+письмо\s+застало\s+вас\s+в\s+добром\s+здравии\b',
     'Шаблонное деловое приветствие; часто используется и людьми'),
    ('importance_caveat', 'medium', r'\bit\s+is\s+(?:important|worth)\s+to\s+note\b',
     'Универсальная вводная конструкция вместо конкретного утверждения'),
    ('importance_caveat_ru', 'medium', r'\b(?:важно|стоит)\s+отметить\b',
     'Универсальная вводная конструкция вместо конкретного утверждения'),
    ('modern_world', 'medium', r"\bin\s+today['’]?s\s+(?:fast-paced\s+)?world\b",
     'Обобщённая шаблонная вводная без привязки к ситуации'),
    ('modern_world_ru', 'medium', r'\bв\s+современном\s+мире\b',
     'Обобщённая шаблонная вводная без привязки к ситуации'),
    ('formulaic_transition', 'weak', r'\b(?:moreover|furthermore|in conclusion|to summarize|to sum up)\b',
     'Формальный переход между тезисами; сам по себе не указывает на ИИ'),
    ('formulaic_transition_ru', 'weak', r'\b(?:кроме того|более того|подводя итог|в заключение)\b',
     'Формальный переход между тезисами; сам по себе не указывает на ИИ'),
    ('generic_verbs', 'weak', r'\b(?:delve into|plays? a crucial role|seamless(?:ly)?|holistic approach|leverage)\b',
     'Распространённая универсальная лексика, которая может делать текст шаблонным'),
    ('generic_verbs_ru', 'weak', r'\b(?:комплексный подход|играет важную роль|позволяет эффективно|способствует развитию)\b',
     'Распространённая универсальная лексика, которая может делать текст шаблонным'),
)

_COMPILED = tuple((key, strength, re.compile(pattern, re.I), explanation)
                  for key, strength, pattern, explanation in _MARKERS)
_TOKEN_RE = re.compile(r"(?u)\b[\w'-]+\b")
_FIRST_PERSON_RE = re.compile(r"(?iu)\b(?:i|we|me|us|my|our|я|мы|мне|нам|мой|моя|моё|наши?)\b")
_CONCRETE_RE = re.compile(r"(?u)\b\d+(?:[.,]\d+)?\s*%?\b|https?://\S+|\b[\w.+-]+@[\w.-]+\.\w+\b", re.I)
_STRENGTH_WEIGHT = {'strong': 1.0, 'medium': 0.55, 'weak': 0.3}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def lexical_zone_features(text: str) -> dict[str, float]:
    """Return bounded, trainable lexical features for one email zone."""
    words = max(1, len(_TOKEN_RE.findall(text)))
    hits = []
    for key, strength, pattern, _ in _COMPILED:
        hits.extend((strength, match.start(), match.end())
                    for match in pattern.finditer(text))
    weighted = sum(_STRENGTH_WEIGHT[strength] for strength, _, _ in hits)
    strong = sum(strength == 'strong' for strength, _, _ in hits)
    first_person = len(_FIRST_PERSON_RE.findall(text))
    concrete = len(_CONCRETE_RE.findall(text))
    scale = words / 50.0
    return {
        'template_marker_rate': _clamp(weighted / max(1.0, scale)),
        'strong_ai_marker_rate': _clamp(strong / max(1.0, scale)),
        'first_person_rate': _clamp(first_person / max(1.0, scale)),
        'concrete_detail_rate': _clamp(concrete / max(1.0, scale)),
    }


def lexical_evidence(zones: dict[str, str]) -> list[dict[str, Any]]:
    """Find exact, verbatim phrase matches and explain their limited meaning."""
    items: list[dict[str, Any]] = []
    for zone in ('subject', 'greeting', 'body', 'signature'):
        text = str(zones.get(zone) or '')
        for key, strength, pattern, explanation in _COMPILED:
            for match in pattern.finditer(text):
                items.append({
                    'zone': zone,
                    'phrase': match.group(0),
                    'start_offset': match.start(),
                    'end_offset': match.end(),
                    'signal': key,
                    'strength': strength,
                    'explanation': explanation,
                    'limitation': 'Это стилевой сигнал, а не доказательство ИИ-авторства.',
                })
    items.sort(key=lambda item: ('subject greeting body signature'.split().index(item['zone']),
                                 item['start_offset']))
    return items
