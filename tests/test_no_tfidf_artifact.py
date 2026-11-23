"""Qwen-only v2 artifacts do not require a dummy tfidf.pkl."""
import json
import pickle

from src.core import analyze
from src.core.scoring import legacy_v2_feature_names, v2_feature_names
from src.api import server


def test_v2_model_loads_without_tfidf_file(monkeypatch, tmp_path):
    (tmp_path / "tree.pkl").write_bytes(pickle.dumps({"no_llm": "dummy-tree"}))
    (tmp_path / "thresholds.json").write_text(json.dumps({"schema_version": 2, "feature_names": v2_feature_names()}), encoding="utf-8")
    (tmp_path / "centroids.json").write_text(json.dumps({"backend": "api"}), encoding="utf-8")
    monkeypatch.setattr(analyze, "MODEL_DIR", tmp_path)
    analyze.load_model.cache_clear()
    try:
        tree, thresholds, centroids = analyze.load_model()
        assert tree == {"no_llm": "dummy-tree"}
        assert thresholds["schema_version"] == 2
        assert centroids["backend"] == "api"
    finally:
        analyze.load_model.cache_clear()


def test_health_accepts_qwen_model_without_tfidf(monkeypatch, tmp_path):
    for name in ("tree.pkl", "thresholds.json", "centroids.json"):
        (tmp_path / name).write_bytes(b"placeholder")
    monkeypatch.setattr(server, "MODEL_DIR", tmp_path)
    assert server.health()["model_ready"] is True


def test_manual_email_input_is_single_line(monkeypatch, capsys):
    import builtins
    from test_my_text import get_text

    message = "Subject: Meeting Request. Hello, please confirm the meeting. Best regards, Alex."
    monkeypatch.setattr(builtins, "input", lambda: message)
    assert get_text(None) == message
    assert "одной строкой" in capsys.readouterr().out


def test_pre_lexical_77_feature_qwen_model_remains_loadable(monkeypatch, tmp_path):
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(max_depth=1, random_state=1).fit(
        np.asarray([[0.] * 77, [1.] * 77]), np.asarray([0, 1]))
    (tmp_path / "tree.pkl").write_bytes(pickle.dumps({"no_llm": tree}))
    (tmp_path / "thresholds.json").write_text(json.dumps({
        "schema_version": 2,
        "feature_names": legacy_v2_feature_names(include_tfidf=True),
    }), encoding="utf-8")
    (tmp_path / "centroids.json").write_text(json.dumps({"backend": "api"}), encoding="utf-8")
    monkeypatch.setattr(analyze, "MODEL_DIR", tmp_path)
    analyze.load_model.cache_clear()
    try:
        loaded, thresholds, _ = analyze.load_model()
        assert loaded["no_llm"].n_features_in_ == 77
        assert len(thresholds["feature_names"]) == 77
    finally:
        analyze.load_model.cache_clear()


def test_legacy_feature_layouts_match_original_dimensions():
    assert len(legacy_v2_feature_names()) == 73
    assert len(legacy_v2_feature_names(include_tfidf=True)) == 77
