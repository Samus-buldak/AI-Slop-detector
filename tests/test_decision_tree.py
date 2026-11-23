import numpy as np

from src.models.decision_tree import train_tuned_tree, tree_feature_importance


def test_tree_training_uses_cv_and_returns_pruned_inspection_metadata():
    rng = np.random.default_rng(12)
    human = rng.normal(loc=-0.6, scale=0.25, size=(60, 4))
    ai = rng.normal(loc=0.6, scale=0.25, size=(60, 4))
    X = np.vstack([human, ai])
    y = np.array([0] * len(human) + [1] * len(ai))

    model, report = train_tuned_tree(X, y, seed=7)
    assert report['selection_method'].startswith('stratified_cv')
    assert report['cv_folds'] == 5
    assert report['actual_depth'] <= 4
    assert report['leaf_count'] >= 2
    assert model.predict(X[:3]).tolist() == [0, 0, 0]
    importance = tree_feature_importance(model, ['f1', 'f2', 'f3', 'f4'])
    assert importance
    assert all(item['importance'] > 0 for item in importance)


def test_tree_rejects_nan_features_instead_of_serializing_a_broken_model():
    X = np.array([[0.0], [1.0], [np.nan], [2.0]])
    y = np.array([0, 0, 1, 1])
    try:
        train_tuned_tree(X, y)
    except ValueError as exc:
        assert 'NaN' in str(exc)
    else:
        raise AssertionError('NaN features must be rejected')
