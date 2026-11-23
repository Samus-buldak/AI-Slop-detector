"""No LM Studio required: test input wiring with deterministic mocked API vectors."""
import json
from email.message import EmailMessage

import numpy as np
import pytest

from src.core import signals_v2
from src.features.context_features import CONTEXT_NAMES, context_features
from src.core.scoring import v2_feature_names, v2_features, scores_for_email
from src.segmentation.segments import KINDS, document_text, parse_eml, split_email


def test_quoted_reply_and_signature_not_misattributed():
    text = ('Hello Michael,\n\nI reviewed the proposal today.\n\n'
            'Best regards,\nAlice\n\nOn Fri, Nov 6, 2026 at 10:23 AM Bob wrote:\n'
            '> This entire old email was written by somebody else.\n')
    s = split_email('Re: Plans', text)
    assert s['greeting'] == 'Hello Michael,'
    assert s['body'] == 'I reviewed the proposal today.'
    assert s['signature'] == 'Best regards,\nAlice'
    assert 'somebody else' not in document_text(s)


def test_html_quoted_thread_and_hidden_data_ignored():
    msg = EmailMessage()
    msg.add_alternative('''<p>Hello!</p><div>New message from me.</div>
        <div class="gmail_quote">On Tuesday Bob wrote: <p>HIDDEN HISTORY</p></div>
        <script>HIDDEN SCRIPT</script>''', subtype='html')
    subject, body = parse_eml(msg.as_bytes())
    assert 'New message from me' in body
    assert 'HIDDEN HISTORY' not in body and 'HIDDEN SCRIPT' not in body
    assert split_email(subject, body)['body'] == 'New message from me.'


def test_subject_and_body_context_features_stable():
    s = split_email('Report', 'Hi there!\n\nA short report.\n\nCheers,\nMary')
    values = context_features(s, 'Hi there!\n\nA short report.\n\nCheers,\nMary')
    assert tuple(values) == CONTEXT_NAMES
    assert all(0 <= x <= 1 for x in values.values())
    assert values['signature_present'] == 1


def test_owned_context_preserves_thread_markers():
    body = ('Here is the update.\n\n----- Forwarded message -----\n'
            '> Old message\n- Old list item')
    zones = split_email('Fwd: Update', body)
    result = signals_v2.collect_rows(
        [('Fwd: Update', body, 0)],
        {'backend': 'offline', 'centroids': {}}, None,
    )[0]['context']
    assert result['has_reply_marker'] == 1
    assert result['has_forward_marker'] == 1
    assert result['quote_line_fraction'] > 0
    assert result['bullet_line_fraction'] == 0


def test_empty_message_has_no_words_per_sentence():
    assert context_features(split_email('', ''))['average_sentence_words'] == 0


def test_llm_is_real_tree_input_not_just_postprocessing():
    zones = {'subject': 'test', 'greeting': '', 'body': 'body', 'signature': ''}
    score = scores_for_email(zones, {'subject': .3, 'body': .4},
                             {'subject': .91, 'body': .92})
    context = context_features(zones, zones['body'])
    base = v2_features(score, context, include_llm=False)
    with_llm = v2_features(score, context, include_llm=True)
    idx = v2_feature_names().index('body_semantic_llm')
    assert len(base) == len(with_llm) == len(v2_feature_names())
    assert base[idx] == 0 and with_llm[idx] == pytest.approx(.92)
    assert base[v2_feature_names().index('body_semantic_available')] == 0
    assert with_llm[v2_feature_names().index('body_semantic_available')] == 1


def test_qwen_embeddings_document_coherence_are_features(monkeypatch):
    async def fake_api(segs, want_embedding=True, want_semantic=True, cache=None, verbose=False):
        emb, sem = {}, {}
        for s in segs:
            for k, text in s.items():
                if text:
                    if want_embedding:
                        emb[(k, text)] = [1., 0.] if 'assistant' in text.lower() else [0., 1.]
                    if want_semantic:
                        sem[(k, text)] = .9 if 'assistant' in text.lower() else .1
        return emb, sem

    def fake_embed(texts, **kwargs):
        return [[1., 0.] if 'assistant' in t.lower() else [0., 1.] for t in texts]

    monkeypatch.setattr(signals_v2, 'api_signals', fake_api)
    monkeypatch.setattr(signals_v2, 'embedding_vectors', fake_embed)
    centers = {'backend': 'api', 'centroids': {
        'document_1': [1., 0.], 'document_0': [0., 1.],
        'subject_1': [1., 0.], 'subject_0': [0., 1.],
        'body_1': [1., 0.], 'body_0': [0., 1.]}, 'model': 'mock-qwen'}
    rows = [('assistant update', 'assistant wrote this', 1),
            ('work update', 'just a quick human note', 0)]
    result = signals_v2.collect_rows(rows, centers, None, with_llm=True)
    assert result[0]['scores']['body']['embedding'] > result[1]['scores']['body']['embedding']
    assert result[0]['scores']['body']['semantic_llm'] == .9
    assert result[0]['doc_signals']['document_ai_similarity'] == pytest.approx(1.)
    assert result[0]['doc_signals']['subject_body_coherence'] == pytest.approx(1.)
    names = v2_feature_names()
    assert result[0]['features_with_llm'][names.index('body_semantic_llm')] == .9
    assert result[0]['features_no_llm'][names.index('body_semantic_llm')] == 0.
    assert result[0]['features_with_llm'][names.index('document_ai_similarity')] == 1.


def test_long_email_embedding_chunks_no_truncation(monkeypatch):
    from src.features import embedding_pipeline
    from src.api.api import _normalise_vector
    text = 'Long message with some details. ' * 300
    parts = embedding_pipeline.chunks(text)
    assert len(parts) > 1 and all(len(x) <= 3000 for x in parts)
    assert parts[-1].endswith('details.')

    async def fake_vectors(inputs, cache=None, verbose=False):
        return [[1., 0.] for _ in inputs]

    monkeypatch.setattr(embedding_pipeline, 'embedding_many', fake_vectors)
    assert embedding_pipeline.embedding_vectors([text, '']) == [[1., 0.], None]


def test_pca_embedding_projection_roundtrip():
    from src.features.embedding_pipeline import fit_document_projection, project_document
    rng = np.random.default_rng(18)
    vectors = rng.normal(size=(30, 12))
    vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).tolist()
    pca = fit_document_projection(vectors)
    assert len(pca['components']) == 8
    assert len(project_document(vectors[0], pca)) == 8
    assert project_document(None, pca) == [None] * 8


def test_html_quote_metadata_and_author_only_context():
    msg = EmailMessage()
    msg.set_content('  ')
    msg.add_alternative(
        '<p>Здравствуйте!</p><ul><li>Мой пункт</li></ul>'
        '<blockquote><span>Старая цитата</span> и продолжение</blockquote>'
        '<div style="display: none">скрытый текст</div>'
        '<p>Новая строка автора.</p>', subtype='html')
    attached = EmailMessage()
    attached.set_content('Текст прикреплённого письма.')
    msg.add_attachment(attached)

    subject, source = parse_eml(msg.as_bytes(), include_html_quotes=True)
    zones = split_email(subject, source)
    context = context_features(zones, source)

    assert '> Старая цитата и продолжение' in source
    assert 'скрытый текст' not in source and 'прикреплённого' not in source
    assert 'Старая цитата' not in document_text(zones)
    assert zones['greeting'] == 'Здравствуйте!'
    assert '- Мой пункт' in zones['body']
    assert 'Новая строка автора.' in zones['body']
    assert context['quote_line_fraction'] == pytest.approx(1 / 4)
    assert context['bullet_line_fraction'] == pytest.approx(1 / 3)
    assert context['has_reply_marker'] == 1


def test_heuristic_preserves_inline_answers_and_from_prose():
    source = ('From: my own notes\n\nOn Bob wrote:\n> Old question\n'
              'My reply.\n| Another quote\nMore of my reply.')
    zones = split_email('', source)
    assert zones['body'] == 'From: my own notes\n\nMy reply.\nMore of my reply.'
    assert context_features(zones, source)['quote_line_fraction'] == pytest.approx(2 / 6)
    assert context_features(zones, 'From: my own notes')['has_reply_marker'] == 0


def test_additional_qwen_zones_leave_context_and_base_features_intact(monkeypatch):
    from src.segmentation.qwen_zones import QWEN_KINDS

    text = ('Hello Kim,\n\nA short introduction.\n\n- First item\n- Second item\n\n'
            'Main discussion.\n\nFinal request.\n\nBest regards,\nAlex\n\nP.S. One more thing.')
    rows = [('Update', text, 0)]
    vectors = []
    def fake_embeddings(texts, **kwargs):
        vectors.extend(texts)
        return [[1., 0.] for _ in texts]
    monkeypatch.setattr(signals_v2, 'embedding_vectors', fake_embeddings)
    centers = {'backend': 'api', 'centroids': {'introduction_1': [1., 0.], 'introduction_0': [0., 1.]}}
    old = signals_v2.collect_rows(rows, centers, None)[0]
    new = signals_v2.collect_rows(rows, centers, None, extra_segments=True)[0]
    zones = new['segments']
    assert tuple(zones) == QWEN_KINDS
    assert zones['introduction'] == 'A short introduction.'
    assert zones['main_text'] == 'Main discussion.'
    assert zones['conclusion'] == 'Final request.'
    assert zones['lists'] == zones['paragraph_2'] == '- First item\n- Second item'
    assert zones['postscript'] == 'P.S. One more thing.'
    assert 'Main discussion.' in vectors and not zones['semantic_block_1']
    assert old['context'] == new['context']
    assert old['features_no_llm'] == new['features_no_llm'][:len(v2_feature_names())]
    names = v2_feature_names(extra_segments=True)
    assert len(names) == len(new['features_no_llm']) == 157
    assert new['features_no_llm'][names.index('introduction_embedding')] > .5


def test_context_counts_url_and_final_sentence_once():
    url = 'Visit https://www.example.com today.'
    assert context_features(split_email('', url), url)['url_count'] == pytest.approx(.2)
    body = 'One sentence. Another sentence'
    assert context_features(split_email('', body), body)['average_sentence_words'] == pytest.approx(.04)
