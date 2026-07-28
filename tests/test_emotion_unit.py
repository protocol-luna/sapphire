import sys
from unittest.mock import MagicMock

sys.modules["fastembed"] = MagicMock()

import pytest
import numpy as np
import tempfile
from pathlib import Path
from sapphire.emotion import (
    _axis_score,
    score_axes,
    load_emotion_examples,
    EmotionState,
)


EMOTION_YAML_CONTENT = """
positive:
  - text: i am happy
    weight: 2
  - text: this is great
negative:
  - text: i am sad
high_arousal:
  - text: i am furious
low_arousal:
  - text: i am calm
"""


class TestAxisScore:
    def test_positive_dominates(self):
        emb = np.array([1.0, 0.0])
        pos = np.array([[1.0, 0.0]])
        neg = np.array([[-1.0, 0.0]])
        score = _axis_score(emb, pos, neg)
        assert score > 0

    def test_negative_dominates(self):
        emb = np.array([-1.0, 0.0])
        pos = np.array([[1.0, 0.0]])
        neg = np.array([[-1.0, 0.0]])
        score = _axis_score(emb, pos, neg)
        assert score < 0

    def test_zero_embedding_returns_zero(self):
        emb = np.array([0.0, 0.0])
        pos = np.array([[1.0, 0.0]])
        neg = np.array([[0.0, 1.0]])
        assert _axis_score(emb, pos, neg) == 0.0

    def test_equal_similarity_returns_near_zero(self):
        emb = np.array([1.0, 1.0]) / np.sqrt(2)
        pos = np.array([[1.0, 1.0]]) / np.sqrt(2)
        neg = np.array([[1.0, 1.0]]) / np.sqrt(2)
        score = _axis_score(emb, pos, neg)
        assert abs(score) < 1e-6


class TestScoreAxes:
    def test_none_embedding_returns_zero(self):
        v, a = score_axes(None, {"positive": np.array([[1.0]])})
        assert v == 0.0
        assert a == 0.0

    def test_none_centroids_returns_zero(self):
        emb = np.array([1.0, 0.0])
        v, a = score_axes(emb, None)
        assert v == 0.0
        assert a == 0.0


class TestLoadEmotionExamples:
    @pytest.fixture
    def yaml_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(EMOTION_YAML_CONTENT)
            path = f.name
        yield path
        Path(path).unlink(missing_ok=True)

    def test_loads_all_poles(self, yaml_file):
        data = load_emotion_examples(yaml_file)
        assert "positive" in data
        assert "negative" in data
        assert "high_arousal" in data
        assert "low_arousal" in data

    def test_weight_expansion(self, yaml_file):
        data = load_emotion_examples(yaml_file)
        assert data["positive"].count("i am happy") == 2


class TestEmotionState:
    def test_unknown_key_returns_zero(self):
        es = EmotionState()
        assert es.get("unknown") == {"valence": 0.0, "arousal": 0.0}

    def test_first_update_applies_signal(self):
        es = EmotionState(decay=0.0)
        s = es.update("conv1", 0.5, 0.3)
        assert s["valence"] == pytest.approx(0.5)
        assert s["arousal"] == pytest.approx(0.3)

    def test_decay_blends_with_previous(self):
        es = EmotionState(decay=0.5)
        es.update("conv1", 0.5, 0.0)
        s = es.update("conv1", 1.0, 0.0)
        val = 0.625  # (0.0 * 0.5 + 0.5 * 0.5) * 0.5 + 1.0 * 0.5
        assert s["valence"] == pytest.approx(val)

    def test_deadzone_zeros_small_signals(self):
        es = EmotionState(decay=0.5, deadzone=0.1)
        s = es.update("conv1", 0.05, 0.03)
        assert s["valence"] == 0.0
        assert s["arousal"] == 0.0

    def test_above_deadzone_passes_through(self):
        es = EmotionState(decay=0.0, deadzone=0.1)
        s = es.update("conv1", 0.15, 0.12)
        assert s["valence"] == pytest.approx(0.15)
        assert s["arousal"] == pytest.approx(0.12)

    def test_different_keys_isolated(self):
        es = EmotionState(decay=0.0)
        es.update("conv_a", 0.8, 0.0)
        es.update("conv_b", -0.5, 0.0)
        assert es.get("conv_a")["valence"] == 0.8
        assert es.get("conv_b")["valence"] == -0.5

    def test_decay_toward_zero_with_no_signal(self):
        es = EmotionState(decay=0.5)
        es.update("conv1", 1.0, 0.0)
        s = es.update("conv1", 0.0, 0.0)
        assert s["valence"] == pytest.approx(0.25)

    @pytest.mark.parametrize("valence,arousal", [(0.0, 0.0), (0.5, -0.3), (-0.8, 0.9), (1.0, 1.0)])
    def test_state_stays_bounded(self, valence, arousal):
        es = EmotionState(decay=0.9)
        for _ in range(20):
            es.update("conv1", valence, arousal)
        s = es.get("conv1")
        assert abs(s["valence"]) <= 1.0
        assert abs(s["arousal"]) <= 1.0
