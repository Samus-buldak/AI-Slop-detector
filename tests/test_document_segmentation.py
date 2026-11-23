from src.segmentation.document import build_document_segmentation


def _by_id(report):
    return {segment['id']: segment for segment in report['segments']}


def test_hierarchical_segmentation_preserves_offsets_and_all_visible_text():
    zones = {
        'subject': 'План работ на 2026 год',
        'greeting': 'Здравствуйте, коллеги!',
        'body': (
            'В 2026 г. мы обновили систему. Результат 🚀 — рост на 12.5%. '
            'Например, задержка снизилась с 2.5 до 1.2 секунды.\n\n'
            'Что сделали:\n'
            '- Обновили сервисы.\n'
            '- Добавили мониторинг и алерты!\n'
            '1. Проверили интеграцию.\n2. Подтвердили результат.\n\n'
            'Итог: система работает стабильнее.'
        ),
        'signature': 'С уважением,\nКоманда',
    }

    report = build_document_segmentation(zones)

    assert report['schema_version'] == '1.0'
    assert report['offset_unit'] == 'unicode_codepoint'
    assert report['offset_scope'] == 'sources[zone]'
    assert report['coverage']['paragraphs']['complete'] is True
    assert report['coverage']['sentences']['complete'] is True
    assert report['coverage']['paragraphs']['coverage'] == 1.0
    assert report['coverage']['sentences']['coverage'] == 1.0

    nodes = _by_id(report)
    assert nodes['subject']['text_preview'] == zones['subject']
    assert nodes['greeting']['text_preview'] == zones['greeting']
    assert nodes['signature']['text_preview'] == zones['signature']
    assert nodes['body.paragraph.2']['content_type'] == 'mixed'

    # Every offset must select exactly the emitted text from its declared zone.
    for segment in report['segments']:
        source = report['sources'][segment['zone']]
        start, end = segment['start_offset'], segment['end_offset']
        assert 0 <= start < end <= len(source)
        assert source[start:end].startswith(segment['text_preview'])

    sentence_text = [report['sources'][node['zone']][node['start_offset']:node['end_offset']]
                     for node in report['segments'] if node['type'] == 'sentence']
    assert any('В 2026 г. мы обновили систему.' in text for text in sentence_text)
    assert any('рост на 12.5%.' in text for text in sentence_text)
    assert any('- Обновили сервисы.' in text for text in sentence_text)
    assert any('- Добавили мониторинг и алерты!' in text for text in sentence_text)
    assert any('1. Проверили интеграцию.' in text for text in sentence_text)
    assert any('2. Подтвердили результат.' in text for text in sentence_text)


def test_empty_and_whitespace_zones_are_safe():
    report = build_document_segmentation({
        'subject': '', 'greeting': '  ', 'body': '\n \n', 'signature': '',
    })
    assert report['segments'] == []
    assert report['counts'] == {'zones': 0, 'paragraphs': 0, 'sentences': 0}
    assert report['coverage']['paragraphs']['complete'] is True
    assert report['coverage']['sentences']['complete'] is True


def test_repeated_semantic_blocks_get_correct_non_overlapping_offsets():
    body = 'Повторяющийся блок.\n\nДругой блок.\n\nПовторяющийся блок.'
    report = build_document_segmentation({
        'subject': '', 'greeting': '', 'body': body, 'signature': '',
        'semantic_block_1': 'Повторяющийся блок.',
        'semantic_block_2': 'Другой блок.',
        'semantic_block_3': 'Повторяющийся блок.',
    }, structural_method='llm_line_ownership')

    assert report['methods']['structural'] == 'llm_line_ownership'
    assert report['methods']['semantic_blocks'] == 'llm'
    blocks = report['semantic_blocks']
    assert [block['text_preview'] for block in blocks] == [
        'Повторяющийся блок.', 'Другой блок.', 'Повторяющийся блок.'
    ]
    assert blocks[0]['start_offset'] < blocks[1]['start_offset'] < blocks[2]['start_offset']
    for block in blocks:
        assert body[block['start_offset']:block['end_offset']].startswith(block['text_preview'])


def test_semantic_block_without_exact_source_match_is_not_given_fake_offsets():
    body = 'Исходный текст документа.'
    report = build_document_segmentation({
        'subject': '', 'greeting': '', 'body': body, 'signature': '',
        'semantic_block_1': 'Переписанный текст, которого в документе нет.',
    })
    assert report['semantic_blocks'] == []


def test_semantic_blocks_from_existing_llm_segmenter_are_supported():
    body = 'Первая мысль.\n\nВторая мысль.'
    report = build_document_segmentation({
        'subject': '', 'greeting': '', 'body': body, 'signature': '',
        '_semantic_blocks': ['Первая мысль.', 'Вторая мысль.'],
    })
    assert [block['text_preview'] for block in report['semantic_blocks']] == [
        'Первая мысль.', 'Вторая мысль.'
    ]
    assert report['methods']['semantic_blocks'] == 'llm'
