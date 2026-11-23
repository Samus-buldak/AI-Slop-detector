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
from src.core.scoring import (final_score_v2,
                     full_semantic_coverage, scores_for_email, v2_feature_names)
from src.segmentation.segments import KINDS, parse_eml, split_email
from src.segmentation.qwen_zones import EXTRA_KINDS, QWEN_KINDS
from src.core.signals_v2 import collect_rows
from src.segmentation.segmentation_llm import segment_many
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
    legacy_names = names.copy()
    for kind in reversed(KINDS):
        legacy_names.insert(legacy_names.index(f'{kind}_embedding'), f'{kind}_tfidf')
    if (thresholds.get('schema_version') != 2 or thresholds.get('use_tfidf') or
            saved_names not in (names, legacy_names, v2_feature_names(extra_segments=True))):
        raise RuntimeError('Несовместимая модель. Переобучи её: python -m src.api.local_qwen train')
    centers = json.loads((MODEL_DIR / 'centroids.json').read_text(encoding='utf-8'))
    # Загружать только доверенные локальные веса.
    with (MODEL_DIR / 'tree.pkl').open('rb') as f:
        trees = pickle.load(f)
    if not isinstance(trees, dict) or 'no_llm' not in trees:
        raise RuntimeError('Несовместимый tree.pkl. Переобучи Qwen-модель.')
    return trees, thresholds, centers


def _tree_path(tree, row, names):
    """Human-readable actual split decisions; not a causal explanation."""
    path = []
    node = 0
    while tree.tree_.children_left[node] != -1:
        index = int(tree.tree_.feature[node])
        threshold = float(tree.tree_.threshold[node])
        value = float(row[index])
        branch = 'left' if value <= threshold else 'right'
        path.append({'feature': names[index], 'value': round(value, 4),
                     'operator': '<=' if branch == 'left' else '>',
                     'threshold': round(threshold, 4)})
        node = (tree.tree_.children_left[node] if branch == 'left'
                else tree.tree_.children_right[node])
    return path


def _score_email_v2(raw: bytes | str, model):
    trees, thresholds, centroids = model
    if centroids['backend'] == 'api' and centroids.get('model') != EMBED_MODEL:
        raise RuntimeError('Qwen embedding model changed: retrain the detector before analysis')
    extra_segments = thresholds['feature_names'] == v2_feature_names(extra_segments=True)
    kinds = QWEN_KINDS if extra_segments else KINDS
    subject, body = parse_eml(raw, include_html_quotes=True)
    requires_llm_segmentation = thresholds.get('llm_segmentation', False)
    use_llm_cleanup = requires_llm_segmentation or os.getenv('DETECTOR_LLM_CLEANUP') == '1'
    compatible = bool(SEMANTIC_API_KEY) and (
        not requires_llm_segmentation or (
            thresholds.get('llm_model') == SEMANTIC_MODEL and
            thresholds.get('llm_endpoint', SEMANTIC_API_URL) == SEMANTIC_API_URL
        )
    )
    if use_llm_cleanup and compatible:
        with SignalCache(MODEL_DIR / 'cache.sqlite3') as seg_cache:
            [(zones, segmentation_ok)] = asyncio.run(segment_many(
                [(subject, body)], cache=seg_cache, semantic_blocks=extra_segments))
    else:
        zones = split_email(subject, body)
        segmentation_ok = False
    present = [k for k in KINDS if zones[k].strip()]
    if not present:
        return {'subject': subject, 'verdict': 'UNCERTAIN', 'reason': 'empty_message',
                'segments': {k: {'present': False, 'score': None} for k in kinds},
                'final_score': None, 'flagged_segments': [], 'warnings': []}
    chat_compatible = ('with_llm' in trees and bool(SEMANTIC_API_KEY) and
                       thresholds.get('llm_model') == SEMANTIC_MODEL and
                       thresholds.get('llm_endpoint', SEMANTIC_API_URL) == SEMANTIC_API_URL)
    with SignalCache(MODEL_DIR / 'cache.sqlite3') as cache:
        rows = collect_rows([(subject, body, 0)], centroids, cache,
                            with_llm=bool(chat_compatible), zone_overrides=[zones],
                            extra_segments=extra_segments)
    item = rows[0]
    scores = item['scores']
    zones = item['segments']
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
    if centroids['backend'] == 'api' and item['doc_signals']['document_embedding'] is None:
        warnings.append('Qwen embedding недоступен или размерность вектора не совпадает. Требуется ручная проверка.')
        label = 'UNCERTAIN'
    if 'with_llm' in trees and mode == 'no_llm':
        warnings.append('Разговорная LLM не настроена, модель/endpoint не совпадают или не все части оценены: используется no_llm.')
    if use_llm_cleanup and not segmentation_ok:
        warnings.append('LLM-очистка недоступна; применена эвристическая сегментация.')
        if requires_llm_segmentation:
            label = 'UNCERTAIN'
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
    return {'subject': subject, 'segments': details,
            'mode': mode, 'schema_version': 2,
            'segmentation_method': ('llm_lines' if segmentation_ok else 'rules'),
            'embedding_backend': centroids['backend'], 'embedding_model': centroids['model'],
            'context_features': item['context'],
            'document_embedding_features': item['doc_signals'],
            'tree_probability': round(p, 5),
            'tree_decision_path': _tree_path(tree, feature_vector, saved_names),
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
    for warning in result.get('warnings', []):
        print('[!] ' + warning)
    print('Оценка вероятностная и не доказывает реальное авторство.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError) as exc:
        sys.exit(f'Ошибка: {exc}')
