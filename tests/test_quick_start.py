import quick_start


def test_matching_chat_args_reuses_exact_local_chat_model(monkeypatch):
    monkeypatch.setattr(quick_start, 'load_config', lambda: {})
    settings = {
        'llm_model': 'qwen-local-chat',
        'llm_endpoint': quick_start.LOCAL_ENDPOINT + '/',
    }
    assert quick_start.matching_chat_args(settings) == ['--llm-model', 'qwen-local-chat']


def test_matching_chat_args_requires_exact_external_identity(monkeypatch):
    monkeypatch.setattr(quick_start, 'load_config', lambda: {
        'API_KEY': 'not-a-real-key',
        'API_URL': 'https://chat.example/v1/',
        'MODEL': 'Qwen3.8-27B',
    })
    settings = {
        'llm_model': 'Qwen3.8-27B',
        'llm_endpoint': 'https://chat.example/v1',
    }
    assert quick_start.matching_chat_args(settings) == ['--llm-from-env']

    settings['llm_model'] = 'different-model'
    assert quick_start.matching_chat_args(settings) is None
    settings['llm_model'] = 'Qwen3.8-27B'
    settings['llm_endpoint'] = 'https://other.example/v1'
    assert quick_start.matching_chat_args(settings) is None


def test_matching_chat_args_rejects_missing_training_identity(monkeypatch):
    monkeypatch.setattr(quick_start, 'load_config', lambda: {})
    assert quick_start.matching_chat_args({}) is None
