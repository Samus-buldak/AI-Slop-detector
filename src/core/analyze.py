"""Analyze emails with v2 Qwen+context+LLM decision trees."""
import argparse
import asyncio
import json
import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.api.api import SignalCache
from src.utils.config import (API_KEY, MODEL_DIR, EMBED_MODEL, MODEL,
                    SEMANTIC_API_KEY, SEMANTIC_API_URL, SEMANTIC_MODEL)
from src.core.scoring import (final_score_v2, full_semantic_coverage,
                     legacy_v2_feature_names, scores_for_email, v2_feature_names)
from src.segmentation.segments import KINDS, RULES_VERSION, parse_eml, split_email
from src.segmentation.qwen_zones import EXTRA_KINDS, QWEN_KINDS
from src.core.signals_v2 import collect_rows
from src.segmentation.segmentation_llm import PROMPT_VERSION, segment_many
from src.segmentation.document import build_document_segmentation
from src.features.lexical_evidence import lexical_evidence
from src.features.llm_evidence import explain_email_evidence
from src.vision.media_images import extract_images
from src.vision.vision import assess_images, combine_scores, IMAGE_WEIGHT

RU = {'subject': 'тема', 'greeting': 'приветствие',
      'body': 'тело', 'signature': 'подпись',
      'introduction': 'вступление', 'main_text': 'основная часть', 'conclusion': 'заключение',
      'lists': 'списки', 'postscript': 'постскриптум'}
RU.update({kind: (f'абзац {kind.rsplit("_", 1)[1]}' if kind.startswith('paragraph_')
                  else f'смысловой блок {kind.rsplit("_", 1)[1]}')
           for kind in EXTRA_KINDS if kind not in RU})


@lru_cache(maxsize=1)
def load_model():
    names = ('tree.pkl', 'thresholds.json', 'centroids.json')
    missing = [name for name in names if not (MODEL_DIR / name).exists()]
    if missing:
        raise RuntimeError('Не найдены модели: ' + ', '.join(missing) + '. Сначала обучи Qwen-модель.')
    thresholds = json.loads((MODEL_DIR / 'thresholds.json').read_text(encoding='utf-8'))
    names = v2_feature_names()
    saved_names = thresholds.get('feature_names')
    legacy_names = (legacy_v2_feature_names(),
                    legacy_v2_feature_names(include_tfidf=True))
    if (thresholds.get('schema_version') != 2 or thresholds.get('use_tfidf') or
            saved_names not in (names, *legacy_names, v2_feature_names(extra_segments=True))):
        raise RuntimeError('Несовместимая модель. Переобучи её: python -m src.api.local_qwen train')
    centers = json.loads((MODEL_DIR / 'centroids.json').read_text(encoding='utf-8'))
    # Загружать только доверенные локальные веса.
    with (MODEL_DIR / 'tree.pkl').open('rb') as f:
        trees = pickle.load(f)
    if not isinstance(trees, dict) or 'no_llm' not in trees:
        raise RuntimeError('Несовместимый tree.pkl. Переобучи Qwen-модель.')
    return trees, thresholds, centers


def _feature_label(name):
    """Map model inputs to explicit Russian descriptions for the report."""
    suffixes = {
        'template_marker_rate': 'плотность шаблонных фраз',
        'strong_ai_marker_rate': 'плотность явных маркеров AI-ответа',
        'first_person_rate': 'частота местоимений первого лица',
        'concrete_detail_rate': 'плотность чисел, дат, ссылок и адресов',
        'embedding': 'оценка сходства эмбеддинга с центроидами',
        'embedding_ai_similarity': 'сходство эмбеддинга с примерами AI',
        'embedding_human_similarity': 'сходство эмбеддинга с примерами человека',
        'semantic_llm': 'оценка chat-модели',
        'present': 'наличие зоны',
        'semantic_available': 'доступность оценки chat-модели',
    }
    zone_labels = RU
    for zone, label in zone_labels.items():
        prefix = zone + '_'
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            if suffix in suffixes:
                return f'{label}: {suffixes[suffix]}'

    context_labels = {
        'log_document_words': 'логарифм количества слов во всём письме',
        'log_body_words': 'логарифм количества слов в теле письма',
        'log_subject_words': 'логарифм количества слов в теме',
        'mean_token_length': 'средняя длина слова',
        'lexical_diversity': 'лексическое разнообразие',
        'average_sentence_words': 'средняя длина предложения',
        'question_per_sentence': 'частота вопросительных знаков',
        'exclamation_per_sentence': 'частота восклицательных знаков',
        'comma_per_word': 'частота запятых относительно числа слов',
        'digit_fraction': 'доля цифр в тексте',
        'upper_fraction': 'доля заглавных букв',
        'newline_per_word': 'частота переносов строк относительно числа слов',
        'bullet_line_fraction': 'доля строк со списочными маркерами',
        'url_count': 'количество ссылок',
        'quote_line_fraction': 'доля строк цитируемой переписки',
        'has_reply_marker': 'наличие маркера ответа/старой переписки',
        'has_forward_marker': 'наличие маркера пересылки',
        'greeting_present': 'наличие приветствия',
        'signature_present': 'наличие подписи',
        'subject_present': 'наличие темы',
        'body_present': 'наличие тела письма',
        'document_embedding': 'сходство всего письма с обучающими центроидами',
        'document_ai_similarity': 'сходство всего письма с AI-примерами',
        'document_human_similarity': 'сходство всего письма с человеческими примерами',
        'subject_body_coherence': 'семантическая согласованность темы и тела',
        'greeting_body_coherence': 'семантическая согласованность приветствия и тела',
        'body_signature_coherence': 'семантическая согласованность тела и подписи',
    }
    if name in context_labels:
        return context_labels[name]
    if name.startswith('document_pca_'):
        return 'компонента семантического представления всего письма'
    return name.replace('_', ' ')


def _tree_path(tree, row, names):
    """Return the actual decision path; it is descriptive, not causal proof."""
    path = []
    node = 0
    while tree.tree_.children_left[node] != -1:
        index = int(tree.tree_.feature[node])
        threshold = float(tree.tree_.threshold[node])
        value = float(row[index])
        branch = 'left' if value <= threshold else 'right'
        feature = names[index]
        operator = '<=' if branch == 'left' else '>'
        path.append({
            'feature': feature,
            'feature_label': _feature_label(feature),
            'value': round(value, 4),
            'operator': operator,
            'threshold': round(threshold, 4),
            'condition': f'{_feature_label(feature)}: {value:.4f} {operator} {threshold:.4f}',
        })
        node = (tree.tree_.children_left[node] if branch == 'left'
                else tree.tree_.children_right[node])
    return path


def _chat_identity_matches(thresholds: dict) -> bool:
    return bool(SEMANTIC_API_KEY) and (
        thresholds.get('llm_model') == SEMANTIC_MODEL and
        thresholds.get('llm_endpoint', SEMANTIC_API_URL) == SEMANTIC_API_URL
    )


def _segmentation_is_compatible(thresholds: dict) -> bool:
    """Check that inference segmentation matches the training feature pipeline.

    Legacy models remain loadable, but a different segmentation version must not
    produce a confident verdict from feature distributions it was not trained on.
    """
    version = thresholds.get('segmentation_version')
    if thresholds.get('llm_segmentation', False):
        return (_chat_identity_matches(thresholds) and version == PROMPT_VERSION)
    return version == RULES_VERSION


def _score_email_v2(raw: bytes | str, model):
    trees, thresholds, centroids = model
    if centroids['backend'] == 'api' and centroids.get('model') != EMBED_MODEL:
        raise RuntimeError('Qwen embedding model changed: retrain the detector before analysis')
    extra_segments = thresholds['feature_names'] == v2_feature_names(extra_segments=True)
    kinds = QWEN_KINDS if extra_segments else KINDS
    subject, body = parse_eml(raw, include_html_quotes=True)
    requires_llm_segmentation = thresholds.get('llm_segmentation', False)
    request_llm_blocks = (requires_llm_segmentation or
                          os.getenv('DETECTOR_LLM_CLEANUP') == '1')
    segmentation_compatible = _segmentation_is_compatible(thresholds)
    zones = split_email(subject, body)
    segmentation_ok = False
    if request_llm_blocks and SEMANTIC_API_KEY:
        # LLM structural boundaries are part of the trained feature pipeline only
        # when llm_segmentation=True. For a rules-trained model, optional LLM
        # enrichment may add semantic blocks to the report, but must never alter
        # the four classification zones used by the existing tree.
        llm_compatible = (segmentation_compatible if requires_llm_segmentation
                          else True)
        if llm_compatible:
            with SignalCache(MODEL_DIR / 'cache.sqlite3') as seg_cache:
                [(llm_zones, llm_used)] = asyncio.run(segment_many(
                    [(subject, body)], cache=seg_cache, semantic_blocks=True))
            if requires_llm_segmentation:
                zones = llm_zones
                segmentation_ok = llm_used
            elif llm_used and llm_zones.get('_semantic_blocks'):
                zones['_semantic_blocks'] = llm_zones['_semantic_blocks']
    present = [k for k in KINDS if zones[k].strip()]
    if not present:
        return {'subject': subject, 'verdict': 'UNCERTAIN', 'reason': 'empty_message',
                'segments': {k: {'present': False, 'score': None} for k in kinds},
                'segmentation': build_document_segmentation(
                    zones, structural_method='llm_line_ownership' if segmentation_ok else 'rules'),
                'textual_evidence': {'method': 'verbatim_phrase_match', 'items': [],
                                     'summary': 'Письмо не содержит анализируемых текстовых зон.'},
                'llm_textual_evidence': {'status': 'not_run', 'items': []},
                'final_score': None, 'flagged_segments': [], 'warnings': []}
    segmentation_zones = zones
    chat_compatible = ('with_llm' in trees and _chat_identity_matches(thresholds) and
                       segmentation_compatible)
    with SignalCache(MODEL_DIR / 'cache.sqlite3') as cache:
        feature_zones = (zones if extra_segments else
                         {kind: zones[kind] for kind in KINDS})
        rows = collect_rows([(subject, body, 0)], centroids, cache,
                            with_llm=bool(chat_compatible), zone_overrides=[feature_zones],
                            extra_segments=extra_segments)
        llm_textual_evidence = asyncio.run(explain_email_evidence(zones, cache=cache))
    item = rows[0]
    scores = item['scores']
    zones = item['segments']
    if segmentation_zones.get('_semantic_blocks'):
        zones = dict(zones)
        zones['_semantic_blocks'] = segmentation_zones['_semantic_blocks']
    segmentation_report = build_document_segmentation(
        zones, structural_method='llm_line_ownership' if segmentation_ok else 'rules')
    mode = ('with_llm' if 'with_llm' in trees and
            full_semantic_coverage(scores) else 'no_llm')
    tree = trees[mode]
    names = v2_feature_names(extra_segments=extra_segments)
    feature_vector = item['features_' + mode]
    saved_names = thresholds['feature_names']
    if saved_names != names:
        # Older Qwen-only trees include four inactive inputs fixed at zero.
        values = dict(zip(names, feature_vector))
        feature_vector = [values.get(name, 0.) for name in saved_names]
    p = float(tree.predict_proba(np.asarray([feature_vector]))[0, 1])
    final = final_score_v2(scores, p, mode)
    cut = thresholds['engine'][mode]
    label = ('HUMAN' if final < cut['human_cut'] else
             'AI' if final >= cut['ai_cut'] else 'UNCERTAIN')
    warnings = []
    if not _segmentation_is_compatible(thresholds):
        expected = PROMPT_VERSION if requires_llm_segmentation else RULES_VERSION
        warnings.append(
            'Версия сегментации в сохранённой модели не совпадает с текущей '
            f'({thresholds.get("segmentation_version")!r} вместо {expected!r}); '
            'переобучи модель перед использованием уверенного вердикта.'
        )
        label = 'UNCERTAIN'
    if centroids['backend'] == 'api' and item['doc_signals']['document_embedding'] is None:
        warnings.append('Qwen embedding недоступен или размерность вектора не совпадает. Требуется ручная проверка.')
        label = 'UNCERTAIN'
    if 'with_llm' in trees and mode == 'no_llm':
        warnings.append('Chat-модель, endpoint или версия сегментации не совпадают с обучением; используется no_llm.')
    if requires_llm_segmentation and not segmentation_ok:
        warnings.append('LLM-сегментация недоступна; применена эвристическая сегментация.')
        label = 'UNCERTAIN'
    elif request_llm_blocks and os.getenv('DETECTOR_LLM_CLEANUP') == '1' and not zones.get('_semantic_blocks'):
        warnings.append('Дополнительные смысловые блоки от LLM недоступны; классификационные зоны не изменены.')
    flagged = []
    details = {}
    for kind in kinds:
        s = scores[kind]
        if not s['present']:
            details[kind] = {'present': False, 'score': None}
            continue
        selected = mode
        if selected == 'with_llm' and s['semantic_llm'] is None:
            selected = 'no_llm'
        threshold = thresholds['segments'].get(kind, {}).get(selected)
        value = s[selected]
        if threshold is None or value is None:
            seg_label = 'UNDETERMINED'
        else:
            seg_label = 'AI' if value >= threshold else 'HUMAN'
        if seg_label == 'AI':
            flagged.append(kind)
        details[kind] = {'present': True, 'text_preview': zones[kind][:300],
                         'embedding': s['embedding'],
                         'embedding_ai_similarity': s['embedding_ai_similarity'],
                         'embedding_human_similarity': s['embedding_human_similarity'],
                         'semantic_llm': s['semantic_llm'], 'score': value,
                         'threshold': threshold, 'label': seg_label}
    confident_labels = {details[k]['label'] for k in KINDS
                        if details[k].get('label') in ('AI', 'HUMAN')}
    if len(confident_labels) == 2 and label != 'UNCERTAIN':
        label = 'UNCERTAIN'
        warnings.append('Части письма противоречат друг другу: возможно смешанное авторство.')
    decision_path = _tree_path(tree, feature_vector, saved_names)
    path_features = {step['feature'] for step in decision_path}
    phrase_evidence = lexical_evidence({kind: zones.get(kind, '') for kind in KINDS})
    for evidence in phrase_evidence:
        evidence['used_by_tree'] = bool(
            f"{evidence['zone']}_template_marker_rate" in path_features or
            (evidence['strength'] == 'strong' and
             f"{evidence['zone']}_strong_ai_marker_rate" in path_features)
        )
    evidence_summary = (
        'Найдены точные шаблонные фразы. Поле used_by_tree показывает, '
        'попал ли соответствующий признак в фактический путь дерева.'
        if phrase_evidence else
        'Шаблонные фразы из словаря не найдены; решение опирается на другие признаки, '
        'например семантическое сходство и структуру.'
    )
    return {'subject': subject, 'segments': details,
            'segmentation': segmentation_report,
            'mode': mode, 'schema_version': 2,
            'segmentation_method': ('llm_lines' if segmentation_ok else 'rules'),
            'embedding_backend': centroids['backend'], 'embedding_model': centroids['model'],
            'context_features': item['context'],
            'document_embedding_features': item['doc_signals'],
            'tree_probability': round(p, 5),
            'tree_decision_path': decision_path,
            'textual_evidence': {
                'method': 'verbatim_phrase_match',
                'items': phrase_evidence,
                'summary': evidence_summary,
                'warning': 'Совпадение фразы не доказывает ИИ-авторство; обычные люди используют те же обороты.',
            },
            'llm_textual_evidence': llm_textual_evidence,
            'final_score': round(float(final), 5), 'engine_thresholds': cut,
            'verdict': label, 'flagged_segments': flagged, 'warnings': warnings,
            'api_key_set': bool(API_KEY),
            'note': 'Scores are heuristic; they cannot establish AI authorship with certainty.'}


def score_email(raw: bytes | str):
    """Text scoring + optional image judgment; no vision API call for text-only mail."""
    result = _score_email_v2(raw, load_model())
    photos, media_warnings = extract_images(raw)
    result.setdefault('warnings', []).extend(media_warnings)
    result['text_final_score'] = result.get('final_score')
    result['text_verdict'] = result.get('verdict')
    result['photos'] = {'count': len(photos), 'checked': 0, 'image_score': None,
                        'weight': IMAGE_WEIGHT if photos else 0., 'items': [],
                        'fusion_calibrated': False}
    if not photos:
        return result
    with SignalCache(MODEL_DIR / 'cache.sqlite3') as cache:
        judgments = asyncio.run(assess_images(photos, cache=cache))
    scores = []
    for photo, judgment in zip(photos, judgments):
        value = judgment['score']
        if value is not None:
            scores.append(value)
        result['photos']['items'].append({
            'filename': photo.filename, 'width': photo.width, 'height': photo.height,
            'score': round(value, 5) if value is not None else None,
            'reason': judgment['reason']})
    result['photos']['checked'] = len(scores)
    if not scores:
        result['warnings'].append('Изображения найдены, но vision-LLM не смогла их оценить; итог по тексту не изменён.')
        return result
    image_score = sum(scores) / len(scores)
    result['photos']['image_score'] = round(image_score, 5)
    text_score = result['text_final_score']
    if text_score is None:
        result['warnings'].append('Текст отсутствует: изображение оценено, но итоговое авторство сообщения определить нельзя.')
        return result
    combined = combine_scores(text_score, scores)
    result['final_score'] = round(combined, 5)
    result['warnings'].append('Оценка изображений — эвристика vision-LLM, вес 20%; итог НЕ откалиброван на фото.')
    cuts = result.get('engine_thresholds')
    if cuts and result.get('verdict') != 'UNCERTAIN':
        predicted = ('HUMAN' if combined < cuts['human_cut'] else
                     'AI' if combined >= cuts['ai_cut'] else 'UNCERTAIN')
        if abs(text_score - image_score) >= 0.55:
            result['verdict'] = 'UNCERTAIN'
            result['warnings'].append('Текст и фото существенно противоречат друг другу.')
        else:
            result['verdict'] = predicted
    return result


def fmt(value):
    return f'{value:.3f}' if value is not None else '   — '


def main():
    parser = argparse.ArgumentParser(description='Детектор AI-авторства .eml')
    parser.add_argument('file', help='путь к письму .eml или - для stdin')
    parser.add_argument('--json', action='store_true', help='машиночитаемый JSON')
    args = parser.parse_args()
    raw = sys.stdin.buffer.read() if args.file == '-' else Path(args.file).read_bytes()
    result = score_email(raw)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    print(f'Тема: {result["subject"] or "(нет)"}')
    print(f'Режим: {result.get("mode", "нет")}; embeddings: {result.get("embedding_backend", "нет")}')
    print('Выделенные части письма:')
    for kind in KINDS:
        segment = result.get('segments', {}).get(kind, {})
        if segment.get('present'):
            preview = ' '.join(segment.get('text_preview', '').split())
            print(f'  {RU[kind]}: {preview[:220]}')
    print(f'{"СЕГМЕНТ":<16} {"EMBED":>9} {"LLM":>9} {"SCORE":>9} {"ПОРОГ":>9} МЕТКА')
    for k, v in result['segments'].items():
        if not v['present']:
            print(f'{RU[k]:<16} (зона отсутствует)')
            continue
        print(f'{RU[k]:<16} {fmt(v["embedding"]):>9} '
              f'{fmt(v["semantic_llm"]):>9} {fmt(v["score"]):>9} '
              f'{fmt(v["threshold"]):>9} {v["label"]}')
    print(f'Decision Tree: {fmt(result.get("tree_probability"))}')
    print(f'Final score: {fmt(result.get("final_score"))}')
    if result.get('photos', {}).get('count'):
        print('Изображений:', result['photos']['count'], 'оценено:', result['photos']['checked'],
              'image_score:', fmt(result['photos']['image_score']))
    print(f'ВЕРДИКТ: {result["verdict"]}')
    for step in result.get('tree_decision_path', []):
        print('  Дерево:', step.get('condition', step.get('feature')))
    for item in result.get('textual_evidence', {}).get('items', []):
        marker = 'использован деревом' if item.get('used_by_tree') else 'не повлиял на путь дерева'
        print(f"  Маркер [{RU.get(item['zone'], item['zone'])}]: «{item['phrase']}» — {marker}")
    llm_evidence = result.get('llm_textual_evidence', {})
    if llm_evidence.get('status') == 'ok':
        print('Пояснение chat-модели:', llm_evidence.get('summary') or '—')
        for item in llm_evidence.get('items', []):
            print(f"  [{RU.get(item['zone'], item['zone'])}] «{item['quote']}» — {item['reason']}")
    for warning in result.get('warnings', []):
        print('[!] ' + warning)
    print('Оценка вероятностная и не доказывает реальное авторство.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError) as exc:
        sys.exit(f'Ошибка: {exc}')
