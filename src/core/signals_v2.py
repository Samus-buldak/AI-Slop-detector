"""Single source of truth for v2 train and inference feature construction."""
from __future__ import annotations

import asyncio

from src.api.api import api_signals
from src.features.embedding_pipeline import embedding_vectors, project_document
from src.features.context_features import context_features
from src.core.scoring import (cosine, embedding_details, scores_for_email, v2_features)
from src.segmentation.segments import KINDS, document_text, split_email
from src.segmentation.qwen_zones import QWEN_KINDS, expand_qwen_zones


def collect_rows(rows, centroids, cache, with_llm=False, verbose=False, zone_overrides=None, *, extra_segments=False, strict_embeddings=False, strict_semantic=False):
    """Rows are (subject, original_body, label). Work is batched and cached."""
    segments = ([zone_overrides[i] if zone_overrides[i] is not None else split_email(subject, body)
                 for i, (subject, body, _) in enumerate(rows)] if zone_overrides is not None else
                [split_email(subject, body) for subject, body, _ in rows])
    kinds = QWEN_KINDS if extra_segments else KINDS
    if extra_segments:
        segments = [expand_qwen_zones(zones) for zones in segments]
    want_embed = centroids['backend'] == 'api'
    if with_llm:
        api_kwargs = {'cache': cache, 'verbose': verbose}
        if strict_semantic:
            api_kwargs['strict_semantic'] = True
        _, semantics = asyncio.run(api_signals(
            [{kind: zones[kind] for kind in KINDS} for zones in segments],
            want_embedding=False, want_semantic=True, **api_kwargs))
    else:
        semantics = {}

    embeddings = {}
    doc_vecs = [None] * len(rows)
    if want_embed:
        zone_keys = list(dict.fromkeys((kind, zones[kind]) for zones in segments
                                      for kind in kinds if zones[kind].strip()))
        doc_texts = [document_text(zones) for zones in segments]
        inputs = [text for kind, text in zone_keys] + doc_texts
        vectors = embedding_vectors(inputs, cache=cache, verbose=verbose, strict=strict_embeddings)
        embeddings = dict(zip(zone_keys, vectors[:len(zone_keys)]))
        doc_vecs = vectors[len(zone_keys):]

    centers = centroids['centroids']
    if strict_embeddings:
        expected_dim = int(centroids.get('dimension') or 0)
        invalid = [kind for (kind, _text), vector in embeddings.items()
                   if vector is not None and expected_dim and len(vector) != expected_dim]
        invalid_docs = [i for i, vector in enumerate(doc_vecs)
                        if vector is None or (expected_dim and len(vector) != expected_dim)]
        if invalid:
            raise RuntimeError('Embedding dimensions changed during training for: ' + ', '.join(sorted(set(invalid))))
        if invalid_docs:
            raise RuntimeError('Missing or dimension-mismatched full-document embeddings for row(s): ' +
                               ', '.join(map(str, invalid_docs[:10])))
    result = []
    for i, ((_, original_body, label), zones) in enumerate(zip(rows, segments)):
        emb_probs, sem_probs, similarities = {}, {}, {}
        for kind in kinds:
            if not zones[kind].strip():
                continue
            vec = embeddings.get((kind, zones[kind]))
            similarities[kind] = embedding_details(
                vec, centers.get(f'{kind}_1'), centers.get(f'{kind}_0'))
            emb_probs[kind] = similarities[kind]['embedding']
            sem_probs[kind] = semantics.get((kind, zones[kind]))
        scores = scores_for_email(zones, emb_probs, sem_probs, similarities)
        doc_detail = embedding_details(doc_vecs[i], centers.get('document_1'),
                                       centers.get('document_0'))
        doc_signals = {'document_embedding': doc_detail['embedding'],
                       'document_ai_similarity': doc_detail['embedding_ai_similarity'],
                       'document_human_similarity': doc_detail['embedding_human_similarity']}
        for n, v in enumerate(project_document(doc_vecs[i], centroids.get('projection'))):
            doc_signals[f'document_pca_{n}'] = v
        for name, a, b in (
            ('subject_body_coherence', 'subject', 'body'),
            ('greeting_body_coherence', 'greeting', 'body'),
            ('body_signature_coherence', 'body', 'signature'),
        ):
            sim = cosine(embeddings.get((a, zones[a])), embeddings.get((b, zones[b])))
            doc_signals[name] = sim
        context = context_features(zones, original_body)
        result.append({'label': label, 'segments': zones, 'scores': scores,
                       'doc_signals': doc_signals, 'context': context,
                       'features_no_llm': v2_features(scores, context, doc_signals, include_llm=False),
                       'features_with_llm': v2_features(scores, context, doc_signals, include_llm=True)})
    return result
