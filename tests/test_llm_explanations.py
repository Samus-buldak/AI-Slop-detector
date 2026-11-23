from src.features.llm_evidence import _validated_evidence, _decode_json
from src.segmentation.segmentation_llm import _parse_boundaries, _apply


def test_chat_explanation_keeps_only_exact_source_quotes_and_offsets():
    zones = {
        'subject': 'Updated project plan',
        'greeting': 'Hello Alex,',
        'body': 'Moreover, the revised plan is ready for review next Tuesday.',
        'signature': 'Best regards,\nSam',
    }
    response = '''Here is my analysis:\n{"summary":"A generic transition appears.","signals":[
      {"zone":"body","quote":"Moreover","type":"formulaic_transition","strength":"weak","reason":"Formal transition"},
      {"zone":"body","quote":"the improved plan is ready","type":"generic_phrase","strength":"moderate","reason":"This is not actually present"},
      {"zone":"greeting","quote":"Hello Alex","type":"other","strength":"weak","reason":"Greeting"}
    ]}'''
    parsed = _validated_evidence(response, zones)
    assert [item['quote'] for item in parsed['items']] == ['Moreover', 'Hello Alex']
    for item in parsed['items']:
        source = zones[item['zone']]
        assert source[item['start_offset']:item['end_offset']] == item['quote']


def test_explanation_json_decoder_accepts_code_fence_and_preamble():
    assert _decode_json('Result:\n```json\n{"signals": []}\n```') == {'signals': []}


def test_llm_segmentation_accepts_preamble_and_uses_integer_boundaries():
    lines = ['Hi Alex,', '', 'The report is ready.', '', 'Regards,', 'Sam']
    response = 'Segmentation:\n' + (
        '[{"i":0,"greeting_end":1,"signature_start":4,"quote_lines":[],"block_starts":[]}]'
    )
    bounds = _parse_boundaries(response, [len(lines)], semantic_blocks=True)[0]
    zones = _apply('Report', lines, bounds, semantic_blocks=True)
    assert zones['subject'] == 'Report'
    assert zones['greeting'] == 'Hi Alex,'
    assert zones['body'] == 'The report is ready.'
    assert zones['signature'] == 'Regards,\nSam'


def test_llm_segmentation_rejects_string_n_instead_of_integer_line_count():
    response = '[{"i":0,"greeting_end":1,"signature_start":"N","quote_lines":[],"block_starts":[]}]'
    try:
        _parse_boundaries(response, [4], semantic_blocks=True)
    except ValueError as exc:
        assert 'boundaries' in str(exc)
    else:
        raise AssertionError('signature_start must be an integer line index')
