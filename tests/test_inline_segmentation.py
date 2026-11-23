from src.segmentation.segments import segmentation_units, split_email
from src.segmentation.segmentation_llm import _apply, _parse_boundaries


SAMPLE = (
    'Hello Michael, We are pleased to announce the launch of our new product line. '
    'Designed with innovation and quality in mind, these products are created to meet '
    'the evolving needs of our customers. What you can expect: Modern solutions for '
    'everyday challenges, Exceptional quality and reliability, A personalized approach '
    'for every customer. You can learn more about our products on our website. '
    'We are always here to help you find the right solution. '
    'Best regards, The Example Company Team'
)


def test_rule_segmentation_splits_greeting_and_final_signature_from_one_line():
    zones = split_email('', SAMPLE)
    assert zones['greeting'] == 'Hello Michael,'
    assert zones['signature'] == 'Best regards, The Example Company Team'
    assert zones['body'].startswith('We are pleased to announce')
    assert zones['body'].endswith('right solution.')
    assert 'Best regards' not in zones['body']


def test_rule_segmentation_does_not_mistake_thank_you_sentence_for_signature():
    body = 'Hello Alex, Thank you for reaching out. We will reply tomorrow.'
    zones = split_email('', body)
    assert zones['greeting'] == 'Hello Alex,'
    assert zones['body'] == 'Thank you for reaching out. We will reply tomorrow.'
    assert not zones['signature']


def test_llm_units_preserve_source_separators_and_allow_single_line_semantic_blocks():
    lines, separators = segmentation_units(SAMPLE)
    assert lines[0] == 'Hello Michael,'
    assert lines[-1] == 'Best regards, The Example Company Team'
    assert len(lines) > 5
    assert len(separators) == len(lines) - 1

    # The model identifies opening greeting, closing sign-off and two semantic
    # block boundaries in the body. Applying these boundaries must not rewrite
    # the original single-line text or inject artificial line breaks.
    greeting_end = 1
    signature_start = len(lines) - 1
    body_units = signature_start - greeting_end
    starts = [greeting_end + max(1, body_units // 2)] if body_units > 2 else []
    parsed = _parse_boundaries(
        '[{"i":0,"quote_lines":[],"greeting_end":%d,"signature_start":%d,"block_starts":%s}]'
        % (greeting_end, signature_start, starts),
        [len(lines)], semantic_blocks=True,
    )[0]
    zones = _apply('', lines, parsed, semantic_blocks=True, separators=separators)

    expected = split_email('', SAMPLE)
    assert zones['greeting'] == expected['greeting']
    assert zones['body'] == expected['body']
    assert zones['signature'] == expected['signature']
    assert ''.join(part + (separators[i] if i < len(separators) else '')
                   for i, part in enumerate(lines)) == SAMPLE
    assert zones['_semantic_blocks']
    assert all(block in expected['body'] for block in zones['_semantic_blocks'])


def test_segment_many_calls_llm_for_sentence_units_and_honors_timeout(monkeypatch):
    import asyncio
    import json

    import httpx

    from src.api import api
    from src.segmentation import segmentation_llm

    monkeypatch.setattr(api, 'SEMANTIC_API_KEY', 'unit-test-token')
    monkeypatch.setattr(api, 'SEMANTIC_API_URL', 'https://llm.example/v1')
    monkeypatch.setattr(segmentation_llm, 'SEMANTIC_API_KEY', 'unit-test-token')
    monkeypatch.setattr(segmentation_llm, 'SEMANTIC_API_URL', 'https://llm.example/v1')
    monkeypatch.setattr(segmentation_llm, 'SEMANTIC_MODEL', 'segmenter-test')
    monkeypatch.setattr(segmentation_llm, 'SEM_BATCH', 1)
    monkeypatch.setattr(segmentation_llm, 'SEM_CONC', 1)
    monkeypatch.setattr(segmentation_llm, 'API_TIMEOUT', 91.0)

    observed = {}
    original_factory = segmentation_llm.create_async_client

    def tracked_factory(*args, **kwargs):
        observed['timeout'] = kwargs.get('timeout')
        return original_factory(*args, **kwargs)

    monkeypatch.setattr(segmentation_llm, 'create_async_client', tracked_factory)

    def handler(request):
        payload = json.loads(request.content)
        data = json.loads(payload['messages'][1]['content'])
        assert request.url.path == '/v1/chat/completions'
        assert request.headers['authorization'] == 'Bearer unit-test-token'
        lines = data[0]['lines']
        assert lines[0] == 'Hello Michael,'
        assert lines[-1] == 'Best regards, The Example Company Team'
        answer = [{
            'i': data[0]['i'],
            'greeting_end': 1,
            'signature_start': len(lines) - 1,
            'quote_lines': [],
            'block_starts': [],
        }]
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(answer)}}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await segmentation_llm.segment_many([('', SAMPLE)], client=client,
                                                        semantic_blocks=True)

    result = asyncio.run(run())
    zones, used = result[0]
    assert observed['timeout'] == 91.0
    assert used is True
    assert zones['greeting'] == 'Hello Michael,'
    assert zones['signature'] == 'Best regards, The Example Company Team'
    assert zones['body'] == split_email('', SAMPLE)['body']


def test_config_numeric_settings_reject_invalid_and_clamp_values(monkeypatch):
    from src.utils import config

    monkeypatch.setenv('TEST_INTEGER_SETTING', 'not-an-int')
    monkeypatch.setenv('TEST_FLOAT_SETTING', 'nan')
    monkeypatch.setenv('TEST_INTEGER_HIGH', '999999')
    assert config._int_setting('TEST_INTEGER_SETTING', 7, 1, 10) == 7
    assert config._float_setting('TEST_FLOAT_SETTING', 9.5, 1.0, 20.0) == 9.5
    assert config._int_setting('TEST_INTEGER_HIGH', 7, 1, 10) == 10


def test_single_content_line_with_trailing_newlines_uses_inline_segmentation():
    expected = split_email('', SAMPLE)
    assert split_email('', SAMPLE + "\n") == expected
    assert split_email('', SAMPLE + "\r\n\r\n") == expected
    plain_units, plain_separators = segmentation_units(SAMPLE)
    crlf_units, crlf_separators = segmentation_units(SAMPLE + "\r\n")
    assert crlf_units == plain_units
    assert crlf_separators == plain_separators


def test_segment_many_splits_invalid_batch_and_recovers_each_email(monkeypatch):
    import asyncio
    import json

    import httpx

    from src.api import api
    from src.segmentation import segmentation_llm

    monkeypatch.setattr(api, 'SEMANTIC_API_KEY', 'unit-test-token')
    monkeypatch.setattr(api, 'SEMANTIC_API_URL', 'https://llm.example/v1')
    monkeypatch.setattr(segmentation_llm, 'SEMANTIC_API_KEY', 'unit-test-token')
    monkeypatch.setattr(segmentation_llm, 'SEMANTIC_API_URL', 'https://llm.example/v1')
    monkeypatch.setattr(segmentation_llm, 'SEMANTIC_MODEL', 'segmenter-test')
    monkeypatch.setattr(segmentation_llm, 'SEM_BATCH', 3)
    monkeypatch.setattr(segmentation_llm, 'SEM_CONC', 1)

    request_sizes = []

    def handler(request):
        payload = json.loads(request.content)
        data = json.loads(payload['messages'][1]['content'])
        request_sizes.append(len(data))
        if len(data) > 1:
            return httpx.Response(200, json={'choices': [{'message': {'content': 'invalid'}}]})
        lines = data[0]['lines']
        answer = [{
            'i': 0,
            'greeting_end': 1 if lines[0].startswith('Hello ') else 0,
            'signature_start': len(lines) - 1 if lines[-1].startswith('Best regards') else len(lines),
            'quote_lines': [],
        }]
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(answer)}}]})

    items = [
        ('', 'Hello A, First body. Best regards, A'),
        ('', 'Hello B, Second body. Best regards, B'),
        ('', 'Hello C, Third body. Best regards, C'),
    ]

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await segmentation_llm.segment_many(items, client=client)

    result = asyncio.run(run())
    assert request_sizes[0] == 3
    assert request_sizes.count(1) == 3
    assert all(used for _, used in result)
    assert [zones['greeting'] for zones, _ in result] == ['Hello A,', 'Hello B,', 'Hello C,']
    assert [zones['signature'] for zones, _ in result] == [
        'Best regards, A', 'Best regards, B', 'Best regards, C']


def test_compound_greeting_keeps_recipient_inside_greeting_zone():
    cases = [
        ('Здравствуйте, коллеги! Просьба проверить отчёт. С уважением, Кирилл',
         'Здравствуйте, коллеги!'),
        ('Hello, everyone! Please review the attached draft. Best regards, Alex',
         'Hello, everyone!'),
        ('Hi, team! The meeting starts at noon.', 'Hi, team!'),
    ]
    for message, expected in cases:
        zones = split_email('', message)
        assert zones['greeting'] == expected
        assert expected not in zones['body']
