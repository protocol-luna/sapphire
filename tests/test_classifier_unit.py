import sys
from unittest.mock import MagicMock

sys.modules["fastembed"] = MagicMock()

import pytest
import numpy as np
import tempfile
from pathlib import Path
from sapphire.classifier import _kmeans, _max_sim, load_examples, classify, save_centroids, load_centroids


EXAMPLES_YAML_CONTENT = """
futile:
  - text: hello
    weight: 2
  - text: hi
interessant:
  - text: what is the meaning of life
    weight: 1
  - text: explain quantum physics
"""


class TestKMeans:
    def test_returns_correct_shape(self):
        data = np.random.rand(50, 384)
        centroids = _kmeans(data, 10)
        assert centroids.shape == (10, 384)

    def test_k_smaller_than_data(self):
        data = np.random.rand(5, 4)
        centroids = _kmeans(data, 10)
        assert centroids.shape == (5, 4)

    def test_converges(self):
        data = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.1], [0.9, 0.9]])
        centroids = _kmeans(data, 2)
        assert centroids.shape == (2, 2)
        assert not np.any(np.isnan(centroids))

    def test_deterministic_with_seed(self):
        data = np.random.rand(100, 10)
        c1 = _kmeans(data, 5)
        c2 = _kmeans(data, 5)
        assert np.allclose(c1, c2)


class TestMaxSim:
    def test_identical_vectors(self):
        emb = np.array([1.0, 0.0, 0.0])
        norm = 1.0
        centroids = np.array([[1.0, 0.0, 0.0]])
        assert _max_sim(emb, norm, centroids) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        emb = np.array([1.0, 0.0])
        norm = 1.0
        centroids = np.array([[-1.0, 0.0]])
        assert _max_sim(emb, norm, centroids) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        emb = np.array([1.0, 0.0])
        norm = 1.0
        centroids = np.array([[0.0, 1.0]])
        sim = _max_sim(emb, norm, centroids)
        assert abs(sim) < 1e-6

    def test_max_over_multiple_centroids(self):
        emb = np.array([1.0, 0.0])
        norm = 1.0
        centroids = np.array([[0.0, 1.0], [1.0, 0.5], [-1.0, 0.0]])
        sim = _max_sim(emb, norm, centroids)
        assert sim > 0.8


class TestClassify:
    @pytest.fixture
    def embedder(self):
        emb = MagicMock()
        emb.query_embed.return_value = iter([np.array([1.0, 0.0])])
        return emb

    def test_futile_with_low_diff(self, embedder):
        f_cent = np.array([[1.0, 0.0]])
        i_cent = np.array([[0.5, 0.5]])
        label, diff, sim_f, sim_i = classify("hello", embedder, f_cent, i_cent)
        assert label == "FUTILE"

    def test_interessant_with_high_diff(self, embedder):
        embedder.query_embed.return_value = iter([np.array([0.0, 1.0])])
        f_cent = np.array([[1.0, 0.0]])
        i_cent = np.array([[0.0, 1.0]])
        label, diff, sim_f, sim_i = classify("hello", embedder, f_cent, i_cent)
        assert label == "INTERESSANT"

    def test_none_centroids_returns_futile(self):
        label, diff, sim_f, sim_i = classify("hello", None, None, None)
        assert label == "FUTILE"
        assert diff == 0.0

    def test_zero_norm_returns_futile(self, embedder):
        embedder.query_embed.return_value = iter([np.array([0.0, 0.0])])
        f_cent = np.array([[1.0, 0.0]])
        i_cent = np.array([[0.0, 1.0]])
        label, diff, sim_f, sim_i = classify("hello", embedder, f_cent, i_cent)
        assert label == "FUTILE"
        assert diff == 0.0

    def test_precomputed_emb_skips_embedder(self):
        emb = np.array([0.0, 1.0])
        f_cent = np.array([[1.0, 0.0]])
        i_cent = np.array([[0.0, 1.0]])
        mock_emb = MagicMock()
        mock_emb.query_embed.side_effect = RuntimeError("should not be called")
        label, diff, sim_f, sim_i = classify("hello", mock_emb, f_cent, i_cent, precomputed_emb=emb)
        assert label == "INTERESSANT"


class TestLoadExamples:
    @pytest.fixture
    def yaml_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(EXAMPLES_YAML_CONTENT)
            path = f.name
        yield path
        Path(path).unlink(missing_ok=True)

    def test_loads_with_weight_expansion(self, yaml_file):
        futile, interessant = load_examples(yaml_file)
        assert len(futile) == 3
        assert futile.count("hello") == 2
        assert len(interessant) == 2

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_examples("/tmp/nonexistent.yml")


class TestSaveLoadCentroids:
    @pytest.fixture
    def npz_path(self):
        path = "/tmp/test_sapphire_centroids.npz"
        yield path
        Path(path).unlink(missing_ok=True)

    def test_save_and_load_roundtrip(self, npz_path):
        f_cent = np.array([[1.0, 0.0], [0.5, 0.5]])
        i_cent = np.array([[0.0, 1.0]])
        save_centroids(f_cent, i_cent, npz_path)
        f_loaded, i_loaded = load_centroids(npz_path)
        assert f_loaded is not None
        assert i_loaded is not None
        assert np.allclose(f_loaded, f_cent)
        assert np.allclose(i_loaded, i_cent)

    def test_load_missing_returns_none(self):
        f, i = load_centroids("/tmp/nonexistent.npz")
        assert f is None
        assert i is None
