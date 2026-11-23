"""No real API or LM Studio needed: all HTTP uses httpx mock transports."""
import asyncio
import io
import json
from email.message import EmailMessage

import httpx
import pytest
from PIL import Image

from src.core import analyze
from src.api import api
from src.segmentation import segmentation_llm
from src.vision import vision
from src.vision.media_images import extract_images
from src.segmentation.segments import split_email


def png_bytes(color='blue'):
    image = Image.new('RGB', (120, 80), color)
    buf = io.BytesIO()
    image.save(buf, 'PNG')
    return buf.getvalue()


def make_email(with_image):
    message = EmailMessage()
    message['Subject'] = 'Привет'
    message.set_content('Привет, коллега\n\nПроверь сообщение.\n\nУдачи,\nОлег')
    if with_image:
        message.add_attachment(png_bytes(), maintype='image', subtype='png', filename='photo.png')
    return message.as_bytes()


def test_images_from_email_only_and_metadata_sanitized():
    raw = make_email(True)
    images, warnings = extract_images(raw)
    assert len(images) == 1 and not warnings
    assert images[0].jpeg.startswith(b'\xff\xd8')
    assert (images[0].width, images[0].height) == (120, 80)
    assert len(extract_images(make_email(False))[0]) == 0


def test_vision_only_when_image_present_and_correct_routing(monkeypatch):
    monkeypatch.setattr(api, 'VISION_API_URL', 'https://vision.example/v1')
    monkeypatch.setattr(api, 'VISION_API_KEY', 'unit-test-token')
    monkeypatch.setattr(vision, 'VISION_API_KEY', 'unit-test-token')
    monkeypatch.setattr(vision, 'VISION_API_URL', 'https://vision.example/v1')
    monkeypatch.setattr(vision, 'VISION_MODEL', 'vision-capable-model')
    images, _ = extract_images(make_email(True))
    calls = []

    def handler(request):
        calls.append(request)
        payload = json.loads(request.content)
        assert request.url.path == '/v1/chat/completions'
        assert request.headers['authorization'] == 'Bearer unit-test-token'
        assert payload['model'] == 'vision-capable-model'
        media = payload['messages'][1]['content'][1]
        assert media['type'] == 'image_url'
        assert media['image_url']['url'].startswith('data:image/jpeg;base64,')
        return httpx.Response(200, json={'choices': [{'message': {'content':
            json.dumps({'assessable': True, 'score': 0.8, 'reason': 'Visual anomalies'})}}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await vision.assess_images([], client=client) == []
            outcome = await vision.assess_images(images, client=client)
        return outcome

    outcome = asyncio.run(run())
    assert len(calls) == 1
    assert outcome[0]['score'] == pytest.approx(.8)
    assert vision.combine_scores(.4, [.8]) == pytest.approx(.48)
    assert vision.combine_scores(.4, []) == pytest.approx(.4)


def test_vision_invalid_json_no_fake_score(monkeypatch):
    monkeypatch.setattr(api, 'VISION_API_URL', 'https://vision.example/v1')
    monkeypatch.setattr(api, 'VISION_API_KEY', 'unit-test-token')
    monkeypatch.setattr(vision, 'VISION_API_KEY', 'unit-test-token')
    monkeypatch.setattr(vision, 'VISION_API_URL', 'https://vision.example/v1')
    monkeypatch.setattr(vision, 'VISION_MODEL', 'vision-capable-model')
    images, _ = extract_images(make_email(True))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={'choices': [{'message': {'content': 'not json'}}]}))) as client:
            return await vision.assess_images(images, client=client)
    assert asyncio.run(run())[0]['score'] is None


def test_llm_segmentation_uses_indices_without_rewriting(monkeypatch):
    monkeypatch.setattr(api, 'SEMANTIC_API_URL', 'https://llm.example/v1')
    monkeypatch.setattr(api, 'SEMANTIC_API_KEY', 'unit-test-token')
    monkeypatch.setattr(segmentation_llm, 'SEMANTIC_API_KEY', 'unit-test-token')
    monkeypatch.setattr(segmentation_llm, 'SEMANTIC_API_URL', 'https://llm.example/v1')
    monkeypatch.setattr(segmentation_llm, 'SEMANTIC_MODEL', 'test-segmenter')
    text = 'Good day Pavel\n\nLong message with details.\nOld text without a quote marker\nNext topic.\n\nYour friend, Alex'
    events = []

    def handler(request):
        payload = json.loads(request.content)
        assert request.url.path == '/v1/chat/completions'
        assert request.headers['authorization'] == 'Bearer unit-test-token'
        data = json.loads(payload['messages'][1]['content'])
        events.append(data)
        answers = [{'i': d['i'], 'greeting_end': 2, 'signature_start': 6, 'quote_lines': [3], 'block_starts': [4]} for d in data]
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(answers)}}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await segmentation_llm.segment_many([('Title', text), ('Title', text)], client=client, semantic_blocks=True)

    result = asyncio.run(run())
    zones, llm_used = result[0]
    assert llm_used and len(events) == 1
    assert result[0] == result[1]
    assert zones['greeting'] == 'Good day Pavel'
    assert zones['body'] == 'Long message with details.\nNext topic.'
    assert zones['_semantic_blocks'] == ['Long message with details.', 'Next topic.']
    assert zones['signature'] == 'Your friend, Alex'
    assert zones['subject'] == 'Title'
    assert 'Long message with details.' in text


def test_bad_segmentation_is_fallback(monkeypatch):
    monkeypatch.setattr(api, 'SEMANTIC_API_URL', 'https://llm.example/v1')
    monkeypatch.setattr(api, 'SEMANTIC_API_KEY', 'unit-test-token')
    monkeypatch.setattr(segmentation_llm, 'SEMANTIC_API_KEY', 'unit-test-token')
    monkeypatch.setattr(segmentation_llm, 'SEMANTIC_MODEL', 'test-segmenter')
    text = 'Hi Pavel\n\nPlease read.\n\nBest regards,\nA.'

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={'choices': [{'message': {'content':
                '[{"i":0,"greeting_end":2,"signature_start":4,"quote_lines":[999]}]'}}]}))) as client:
            return await segmentation_llm.segment_many([('Subject', text)], client=client)
    zones, used = asyncio.run(run())[0]
    assert not used
    assert zones == split_email('Subject', text)


def test_combination_in_inference_no_image_does_not_call_vision(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze, 'MODEL_DIR', tmp_path)
    monkeypatch.setattr(analyze, 'load_model', lambda: ({'no_llm': object()}, {'schema_version': 2}, {}))
    monkeypatch.setattr(analyze, '_score_email_v2', lambda raw, model: {
        'final_score': .6, 'verdict': 'AI', 'engine_thresholds': {
            'human_cut': .25, 'ai_cut': .55}, 'warnings': [], 'segments': {}})
    calls = []

    async def fake_vision(images, cache):
        calls.append(len(images))
        return [{'score': .9, 'reason': 'test'} for _ in images]

    monkeypatch.setattr(analyze, 'assess_images', fake_vision)
    no_image = analyze.score_email(make_email(False))
    assert calls == []
    assert no_image['final_score'] == .6
    with_image = analyze.score_email(make_email(True))
    assert calls == [1]
    assert with_image['photos']['count'] == 1
    assert with_image['photos']['checked'] == 1
    assert with_image['final_score'] == pytest.approx(.66)
    assert with_image['text_final_score'] == .6
    assert with_image['photos']['fusion_calibrated'] is False


def test_analysis_response_contains_hierarchical_segmentation_without_changing_features(
        monkeypatch, tmp_path):
    import numpy as np
    from src.core.scoring import scores_for_email, v2_feature_names, v2_features
    from src.segmentation.segments import KINDS, RULES_VERSION

    class LeafTree:
        class TreeStructure:
            children_left = np.array([-1])
            children_right = np.array([-1])
            feature = np.array([-2])
            threshold = np.array([-2.0])
        tree_ = TreeStructure()

        def predict_proba(self, _features):
            return np.array([[0.2, 0.8]])

    def fake_collect_rows(rows, centroids, cache, with_llm=False, verbose=False,
                          zone_overrides=None, *, extra_segments=False):
        subject, body, label = rows[0]
        zones = zone_overrides[0]
        scores = scores_for_email(zones, {}, {})
        context = {}
        doc_signals = {}
        return [{
            'label': label,
            'segments': zones,
            'scores': scores,
            'context': context,
            'doc_signals': doc_signals,
            'features_no_llm': v2_features(scores, context, doc_signals, False),
            'features_with_llm': v2_features(scores, context, doc_signals, True),
        }]

    monkeypatch.setattr(analyze, 'collect_rows', fake_collect_rows)
    monkeypatch.setattr(analyze, 'MODEL_DIR', tmp_path)
    monkeypatch.delenv('DETECTOR_LLM_CLEANUP', raising=False)

    thresholds = {
        'schema_version': 2,
        'feature_names': v2_feature_names(),
        'llm_segmentation': False,
        'segmentation_version': RULES_VERSION,
        'engine': {'no_llm': {'human_cut': 0.25, 'ai_cut': 0.55}},
        'segments': {kind: {'no_llm': 0.5} for kind in KINDS},
    }
    centroids = {'backend': 'test', 'model': 'test', 'centroids': {}}
    model = ({'no_llm': LeafTree()}, thresholds, centroids)
    raw = (
        b'Subject: Update\nContent-Type: text/plain; charset=utf-8\n\n'
        b'Hello team,\n\nFirst sentence. Second sentence!\n\n'
        b'- Item one.\n- Item two.\n\nRegards,\nAlex'
    )

    result = analyze._score_email_v2(raw, model)
    report = result['segmentation']

    assert report['counts'] == {'zones': 4, 'paragraphs': 2, 'sentences': 4}
    assert report['coverage']['paragraphs']['complete']
    assert report['coverage']['sentences']['complete']
    assert result['segmentation_method'] == 'rules'
    assert result['verdict'] in {'AI', 'HUMAN', 'UNCERTAIN'}
    assert thresholds['feature_names'] == v2_feature_names()
    for segment in report['segments']:
        source = report['sources'][segment['zone']]
        assert source[segment['start_offset']:segment['end_offset']].startswith(
            segment['text_preview'])


def test_llm_semantic_blocks_are_reported_but_not_injected_into_legacy_features(
        monkeypatch, tmp_path):
    import numpy as np
    from src.core.scoring import scores_for_email, v2_feature_names, v2_features
    from src.segmentation.segments import KINDS, RULES_VERSION, split_email

    class LeafTree:
        class TreeStructure:
            children_left = np.array([-1])
            children_right = np.array([-1])
            feature = np.array([-2])
            threshold = np.array([-2.0])
        tree_ = TreeStructure()

        def predict_proba(self, _features):
            return np.array([[0.2, 0.8]])

    async def fake_segment_many(items, **kwargs):
        subject, body = items[0]
        zones = split_email(subject, body)
        zones['greeting'] = 'LLM-ONLY-REPORT-BOUNDARY'
        zones['_semantic_blocks'] = ['First sentence.', 'Second sentence!']
        return [(zones, True)]

    def fake_collect_rows(rows, centroids, cache, with_llm=False, verbose=False,
                          zone_overrides=None, *, extra_segments=False):
        zones = zone_overrides[0]
        assert '_semantic_blocks' not in zones  # report-only enrichment is not a model feature
        assert zones['greeting'] == 'Hello team,'  # no_llm classifier keeps rule boundaries
        scores = scores_for_email(zones, {}, {})
        context = {}
        doc_signals = {}
        return [{
            'label': rows[0][2],
            'segments': zones,
            'scores': scores,
            'context': context,
            'doc_signals': doc_signals,
            'features_no_llm': v2_features(scores, context, doc_signals, False),
            'features_with_llm': v2_features(scores, context, doc_signals, True),
        }]

    monkeypatch.setattr(analyze, 'collect_rows', fake_collect_rows)
    monkeypatch.setattr(analyze, 'segment_many', fake_segment_many)
    monkeypatch.setattr(analyze, 'MODEL_DIR', tmp_path)
    monkeypatch.setattr(analyze, 'SEMANTIC_API_KEY', 'test-key')
    monkeypatch.setenv('DETECTOR_LLM_CLEANUP', '1')

    thresholds = {
        'schema_version': 2,
        'feature_names': v2_feature_names(),
        'llm_segmentation': False,
        'segmentation_version': RULES_VERSION,
        'engine': {'no_llm': {'human_cut': 0.25, 'ai_cut': 0.55}},
        'segments': {kind: {'no_llm': 0.5} for kind in KINDS},
    }
    centroids = {'backend': 'test', 'model': 'test', 'centroids': {}}
    raw = (
        b'Subject: Update\nContent-Type: text/plain; charset=utf-8\n\n'
        b'Hello team,\n\nFirst sentence. Second sentence!\n\nRegards,\nAlex'
    )

    result = analyze._score_email_v2(raw, ({'no_llm': LeafTree()}, thresholds, centroids))
    report = result['segmentation']

    assert result['segmentation_method'] == 'rules'
    assert [block['text_preview'] for block in report['semantic_blocks']] == [
        'First sentence.', 'Second sentence!'
    ]
    assert report['methods']['semantic_blocks'] == 'llm'
    assert thresholds['feature_names'] == v2_feature_names()


def test_segmentation_parser_accepts_json_preamble_but_requires_integer_fallback_boundary():
    parsed = segmentation_llm._parse_boundaries(
        'Результат:\n[{"i":0,"quote_lines":[],"greeting_end":1,"signature_start":4}]',
        [4],
    )
    assert parsed == [(1, 4, [], [])]
    with pytest.raises(ValueError):
        segmentation_llm._parse_boundaries(
            '[{"i":0,"quote_lines":[],"greeting_end":0,"signature_start":"N"}]',
            [4],
        )


def test_rule_segmentation_model_requires_matching_rules_version():
    from src.segmentation.segments import RULES_VERSION

    assert analyze._segmentation_is_compatible({
        'llm_segmentation': False, 'segmentation_version': RULES_VERSION,
    }) is True
    assert analyze._segmentation_is_compatible({
        'llm_segmentation': False, 'segmentation_version': 'rules',
    }) is False
    assert analyze._segmentation_is_compatible({
        'llm_segmentation': False,
    }) is False


def test_llm_segmentation_model_requires_matching_prompt_version(monkeypatch):
    from src.segmentation.segmentation_llm import PROMPT_VERSION

    monkeypatch.setattr(analyze, 'SEMANTIC_API_KEY', 'test-token')
    monkeypatch.setattr(analyze, 'SEMANTIC_MODEL', 'qwen-chat')
    monkeypatch.setattr(analyze, 'SEMANTIC_API_URL', 'https://chat.example/v1')

    base = {
        'llm_segmentation': True,
        'llm_model': 'qwen-chat',
        'llm_endpoint': 'https://chat.example/v1',
    }
    assert analyze._segmentation_is_compatible({
        **base, 'segmentation_version': PROMPT_VERSION,
    }) is True
    assert analyze._segmentation_is_compatible({
        **base, 'segmentation_version': 'line-ownership-v4',
    }) is False
    assert analyze._segmentation_is_compatible({
        **base, 'segmentation_version': None,
    }) is False

    monkeypatch.setattr(analyze, 'SEMANTIC_MODEL', 'different-chat-model')
    assert analyze._segmentation_is_compatible({
        **base, 'segmentation_version': PROMPT_VERSION,
    }) is False
