import csv
import json
import pickle

import pytest

from src.core import train


def test_load_data_excludes_entire_conflicting_duplicate_group(tmp_path):
    path = tmp_path / 'dataset.csv'
    rows = []
    for label in (0, 1):
        for i in range(21):
            rows.append({'subject': f'{label}-{i}', 'text': f'unique message {label} {i}',
                         'is_ai': str(label)})
    rows.extend([
        {'subject': 'duplicate', 'text': 'same   message', 'is_ai': '0'},
        {'subject': ' DUPLICATE ', 'text': 'same message ', 'is_ai': '1'},
        {'subject': 'duplicate', 'text': 'same message', 'is_ai': '0'},
        {'subject': 'repeat', 'text': 'same class text', 'is_ai': '1'},
        {'subject': ' repeat ', 'text': 'same   class text ', 'is_ai': '1'},
    ])
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['subject', 'text', 'is_ai'])
        writer.writeheader()
        writer.writerows(rows)

    loaded = train.load_data(path)
    assert len(loaded) == 43
    assert sum(label == 0 for _, _, label in loaded) == 21
    assert sum(label == 1 for _, _, label in loaded) == 22
    assert not any(subject.strip().casefold() == 'duplicate' for subject, _, _ in loaded)
    assert sum(subject.strip().casefold() == 'repeat' for subject, _, _ in loaded) == 1


def test_split_rows_rejects_non_positive_partition_sizes():
    rows = [(str(i), f'human {i}', 0) for i in range(30)]
    rows += [(str(i), f'ai {i}', 1) for i in range(30)]
    with pytest.raises(ValueError, match='положительными'):
        train.split_rows(rows, 42, 0, 5, 5)


def test_save_artifacts_publishes_a_complete_model_set(tmp_path):
    trees = {'no_llm': {'weight': 1}}
    centroids = {'model': 'test-embed'}
    thresholds = {'schema_version': 2, 'feature_names': ['a']}
    metrics = {'no_llm': {'roc_auc': 0.75}}

    train.save_artifacts(tmp_path, trees, centroids, thresholds, metrics)

    with (tmp_path / 'tree.pkl').open('rb') as handle:
        assert pickle.load(handle) == trees
    assert json.loads((tmp_path / 'centroids.json').read_text()) == centroids
    assert json.loads((tmp_path / 'thresholds.json').read_text()) == thresholds
    assert json.loads((tmp_path / 'test_metrics.json').read_text()) == metrics
    assert not list(tmp_path.glob('.training-stage-*'))


def test_save_artifacts_restores_previous_model_if_publish_fails(tmp_path, monkeypatch):
    names = ('tree.pkl', 'centroids.json', 'thresholds.json', 'test_metrics.json')
    for name in names:
        (tmp_path / name).write_bytes(('old:' + name).encode())
    original_replace = train.os.replace
    state = {'calls': 0, 'failed': False}

    def fail_once(source, destination):
        state['calls'] += 1
        if state['calls'] == 3 and not state['failed']:
            state['failed'] = True
            raise OSError('simulated disk error')
        return original_replace(source, destination)

    monkeypatch.setattr(train.os, 'replace', fail_once)
    with pytest.raises(OSError, match='simulated disk error'):
        train.save_artifacts(tmp_path, {'no_llm': object()}, {}, {}, {})

    for name in names:
        assert (tmp_path / name).read_bytes() == ('old:' + name).encode()
    assert not list(tmp_path.glob('.training-stage-*'))


def test_train_cli_rejects_conflicting_no_semantic_and_llm_segmentation(monkeypatch):
    import sys

    monkeypatch.setattr(train, 'API_KEY', 'test-embedding-key')
    monkeypatch.setattr(sys, 'argv', [
        'train.py', '--no-semantic', '--llm-segmentation',
    ])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2


def test_score_rows_keeps_rule_and_llm_segmentation_profiles_separate(monkeypatch):
    rows = [('Meeting', 'Hello team. Main text. Regards, Alex', 0)]
    zones = {
        'subject': 'Meeting', 'greeting': 'Hello team.',
        'body': 'Main text.', 'signature': 'Regards, Alex',
    }
    captured = []

    def collect(*args, **kwargs):
        captured.append(kwargs)
        return [{'label': 0, 'features_no_llm': [0.0]}]

    async def successful_segment(items, **kwargs):
        assert items == [('Meeting', 'Hello team. Main text. Regards, Alex')]
        return [(zones, True)]

    monkeypatch.setattr(train, 'collect_rows', collect)
    monkeypatch.setattr(train, 'segment_many', successful_segment)

    train.score_rows(rows, {}, None, use_sem=False, llm_segment=False)
    assert captured[-1]['with_llm'] is False
    assert captured[-1]['zone_overrides'] is None
    assert captured[-1]['strict_embeddings'] is True

    train.score_rows(rows, {}, None, use_sem=True, llm_segment=True)
    assert captured[-1]['with_llm'] is True
    assert captured[-1]['zone_overrides'] == [zones]
    assert captured[-1]['strict_semantic'] is True

    async def failed_segment(_items, **_kwargs):
        return [(zones, False)]

    monkeypatch.setattr(train, 'segment_many', failed_segment)
    with pytest.raises(RuntimeError, match='prevent inconsistent features'):
        train.score_rows(rows, {}, None, use_sem=False, llm_segment=True)


class _TrainingSmokeTree:
    def predict_proba(self, features):
        import numpy as np
        return np.tile(np.array([[0.25, 0.75]]), (len(features), 1))


def test_training_pipeline_smoke_for_no_llm_and_full_llm_profiles(tmp_path, monkeypatch):
    import sys

    import numpy as np
    from src.core.scoring import v2_feature_names
    from src.segmentation.segmentation_llm import PROMPT_VERSION
    from src.segmentation.segments import KINDS, RULES_VERSION

    class Cache:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    rows = [(f'subject {label}-{i}', f'message {label}-{i}', label)
            for label in (0, 1) for i in range(100)]
    feature_count = len(v2_feature_names(extra_segments=True))
    calls = []
    centroid_calls = []

    def fake_fit_centroids(*_args, **kwargs):
        centroid_calls.append(kwargs.get('llm_segment', False))
        return {
            'backend': 'api', 'model': 'qwen3-embedding-local', 'dimension': 2,
            'centroids': {},
        }

    def fake_score_rows(part, _centroids, _cache, use_sem, llm_segment=False):
        calls.append((use_sem, llm_segment, len(part)))
        output = []
        for subject, body, label in part:
            scores = {
                kind: {'present': True, 'semantic_llm': 0.6 if use_sem else None,
                       'no_llm': 0.4, 'with_llm': 0.5}
                for kind in KINDS
            }
            output.append({
                'label': label,
                'scores': scores,
                'features_no_llm': np.zeros(feature_count),
                'features_with_llm': np.ones(feature_count),
            })
        return output

    monkeypatch.setattr(train, 'API_KEY', 'embedding-key')
    monkeypatch.setattr(train, 'SEMANTIC_API_KEY', 'chat-key')
    monkeypatch.setattr(train, 'SEMANTIC_API_URL', 'https://chat.example/v1')
    monkeypatch.setattr(train, 'SEMANTIC_MODEL', 'chat-model')
    monkeypatch.setattr(train, 'load_data', lambda _path=None: rows)
    monkeypatch.setattr(train, 'SignalCache', lambda _path: Cache())
    monkeypatch.setattr(train, 'fit_centroids', fake_fit_centroids)
    monkeypatch.setattr(train, 'score_rows', fake_score_rows)
    monkeypatch.setattr(train, 'train_tuned_tree', lambda *_args, **_kwargs: (
        _TrainingSmokeTree(), {'selection_method': 'test'}))
    monkeypatch.setattr(train, 'export_text', lambda *_args, **_kwargs: 'test tree')
    monkeypatch.setattr(train, 'tree_feature_importance', lambda *_args, **_kwargs: [])
    monkeypatch.setattr(train, 'calibrate_segments', lambda _rows: {})
    monkeypatch.setattr(train, 'engine_thresholds', lambda *_args, **_kwargs: {
        'human_cut': 0.3, 'ai_cut': 0.7, 'youden_cut': 0.5, 'youden_j': 0.5,
        'n_calibration': 20,
    })
    monkeypatch.setattr(train, 'evaluate', lambda *_args, **_kwargs: {'smoke': {'ok': True}})

    no_llm_dir = tmp_path / 'no_llm'
    monkeypatch.setattr(train, 'MODEL_DIR', no_llm_dir)
    monkeypatch.setattr(sys, 'argv', ['train.py'])
    train.main()
    no_llm_thresholds = json.loads((no_llm_dir / 'thresholds.json').read_text())
    assert no_llm_thresholds['with_llm_trained'] is False
    assert no_llm_thresholds['llm_segmentation'] is False
    assert no_llm_thresholds['segmentation_version'] == RULES_VERSION
    assert centroid_calls[-1] is False
    assert calls[-3:] and all(not use_sem and not llm_seg for use_sem, llm_seg, _ in calls[-3:])

    llm_dir = tmp_path / 'full_llm'
    monkeypatch.setattr(train, 'MODEL_DIR', llm_dir)
    monkeypatch.setattr(sys, 'argv', ['train.py', '--with-llm', '--llm-segmentation'])
    train.main()
    llm_thresholds = json.loads((llm_dir / 'thresholds.json').read_text())
    assert llm_thresholds['with_llm_trained'] is True
    assert llm_thresholds['llm_segmentation'] is True
    assert centroid_calls[-1] is True
    assert llm_thresholds['segmentation_version'] == PROMPT_VERSION
    assert llm_thresholds['llm_model'] == 'chat-model'
    assert llm_thresholds['llm_endpoint'] == 'https://chat.example/v1'
    assert all(use_sem and llm_seg for use_sem, llm_seg, _ in calls[-3:])
    assert (llm_dir / 'tree.pkl').is_file()
    assert (llm_dir / 'centroids.json').is_file()
    assert (llm_dir / 'test_metrics.json').is_file()
