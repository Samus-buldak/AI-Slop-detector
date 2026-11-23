from src.features.context_features import CONTEXT_NAMES, context_features
from src.features.lexical_evidence import lexical_evidence, lexical_zone_features
from src.segmentation.segments import KINDS


def test_phrase_evidence_is_verbatim_and_has_exact_offsets():
    zones = {
        'subject': 'Project update',
        'greeting': 'Hello team,',
        'body': 'Moreover, it is important to note that delivery is complete.',
        'signature': 'Regards,\nAlex',
    }
    found = lexical_evidence(zones)
    phrases = [item['phrase'] for item in found]
    assert 'Moreover' in phrases
    assert 'it is important to note' in phrases
    for item in found:
        source = zones[item['zone']]
        assert source[item['start_offset']:item['end_offset']] == item['phrase']
        assert item['limitation']


def test_lexical_features_are_bounded_and_zone_specific():
    weak = lexical_zone_features('Moreover, it is important to note the following.')
    neutral = lexical_zone_features('I sent the report on 2026-11-23 with 3 results.')
    assert 0 < weak['template_marker_rate'] <= 1
    assert neutral['template_marker_rate'] == 0
    assert 0 <= neutral['first_person_rate'] <= 1
    assert neutral['concrete_detail_rate'] > 0


def test_context_features_expose_zone_level_lexical_inputs():
    zones = {'subject': 'Update', 'greeting': 'Hello',
             'body': 'Furthermore, the task is complete.', 'signature': 'Regards'}
    features = context_features(zones, '\n'.join(zones.values()))
    for zone in KINDS:
        assert f'{zone}_template_marker_rate' in features
        assert f'{zone}_strong_ai_marker_rate' in features
    assert set(features) == set(CONTEXT_NAMES)
    assert features['body_template_marker_rate'] > 0
    assert features['subject_template_marker_rate'] == 0
