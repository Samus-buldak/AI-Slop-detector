"""Small, pruned and inspectable decision trees for the email detector."""
from __future__ import annotations

import pickle
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_tuned_tree(X, y, *, seed: int = 42) -> tuple[DecisionTreeClassifier, dict[str, Any]]:
    """Select tree complexity with stratified CV on the stack split only.

    Calibration and test partitions are deliberately not used here. The
    candidate set favours shallow trees and post-pruning; the chosen estimator
    is then refitted on the complete stack split.
    """
    features = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    if features.ndim != 2 or features.shape[0] != labels.shape[0] or not len(labels):
        raise ValueError('Decision tree expects a non-empty 2D feature matrix and matching labels')
    if not np.isfinite(features).all():
        raise ValueError('Decision-tree features contain NaN or infinite values')
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2 or counts.min() < 2:
        raise ValueError('Decision-tree training requires at least two examples of each class')

    folds = min(5, int(counts.min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    n = len(labels)
    min_class = int(counts.min())
    leaf_candidates = sorted({
        max(2, min(min_class // 2, max(2, n // 100))),
        max(2, min(min_class // 2, max(2, n // 60))),
        max(2, min(min_class // 2, max(3, n // 35))),
    })
    alpha_candidates = (0.0, 0.0005, 0.001, 0.003, 0.01)
    depth_candidates = (2, 3, 4)

    candidates = []
    for depth in depth_candidates:
        for leaf in leaf_candidates:
            for alpha in alpha_candidates:
                estimator = DecisionTreeClassifier(
                    criterion='gini', max_depth=depth, min_samples_leaf=leaf,
                    class_weight='balanced', ccp_alpha=alpha, random_state=seed,
                )
                scores = cross_val_score(estimator, features, labels, cv=cv,
                                         scoring='balanced_accuracy', n_jobs=1,
                                         error_score='raise')
                candidates.append({
                    'mean_balanced_accuracy': float(np.mean(scores)),
                    'std_balanced_accuracy': float(np.std(scores)),
                    'max_depth': depth,
                    'min_samples_leaf': leaf,
                    'ccp_alpha': alpha,
                })

    # Scores within 0.002 are treated as practically tied; prefer the simpler
    # tree instead of chasing tiny validation fluctuations.
    best_score = max(item['mean_balanced_accuracy'] for item in candidates)
    eligible = [item for item in candidates
                if item['mean_balanced_accuracy'] >= best_score - 0.002]
    chosen = max(eligible, key=lambda item: (
        item['max_depth'] * -1,
        item['min_samples_leaf'],
        item['ccp_alpha'],
    ))
    model = DecisionTreeClassifier(
        criterion='gini', max_depth=chosen['max_depth'],
        min_samples_leaf=chosen['min_samples_leaf'],
        class_weight='balanced', ccp_alpha=chosen['ccp_alpha'], random_state=seed,
    ).fit(features, labels)
    metadata = {
        'selection_method': 'stratified_cv_balanced_accuracy_with_simplicity_tie_break',
        'cv_folds': folds,
        'best_cv_balanced_accuracy': round(chosen['mean_balanced_accuracy'], 5),
        'cv_std': round(chosen['std_balanced_accuracy'], 5),
        'parameters': {key: chosen[key] for key in
                       ('max_depth', 'min_samples_leaf', 'ccp_alpha')},
        'actual_depth': int(model.get_depth()),
        'node_count': int(model.tree_.node_count),
        'leaf_count': int(model.get_n_leaves()),
    }
    logger.info('Decision tree trained: depth=%s, leaves=%s, CV balanced accuracy=%.4f',
                metadata['actual_depth'], metadata['leaf_count'],
                metadata['best_cv_balanced_accuracy'])
    return model, metadata


def tree_feature_importance(model, feature_names: list[str], top_n: int = 20) -> list[dict[str, Any]]:
    importances = getattr(model, 'feature_importances_', None)
    if importances is None or len(importances) != len(feature_names):
        return []
    order = np.argsort(importances)[::-1]
    return [
        {'feature': feature_names[int(index)], 'importance': round(float(importances[index]), 6)}
        for index in order[:top_n] if importances[index] > 0
    ]


class ModelManager:
    """Compatibility wrapper for older callers."""

    @staticmethod
    def train(X_train, y_train, max_depth=3):
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42,
                                     class_weight='balanced', min_samples_leaf=2)
        clf.fit(X_train, y_train)
        logger.info('Дерево решений обучено.')
        return clf

    @staticmethod
    def save(model, path: str):
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info('Модель сохранена в %s', path)

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)
