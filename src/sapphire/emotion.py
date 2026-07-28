"""
Sapphire emotion axes -- continuous valence/arousal via multicentroid similarity.

Same mechanism as classifier.py (k-means multicentroid + max cosine similarity),
but instead of a discrete label, each axis returns a continuous score in [-1, 1]:
signed difference of max cosine similarity to the two poles of the axis.

No new embedding call needed at request time if you reuse the same
embedding computed for the FUTILE/INTERESSANT classification -- see
server.py's chat_completions handler, where `emb` is computed once and
passed to both classify() and score_axes().
"""

from pathlib import Path

import numpy as np
import yaml
from fastembed import TextEmbedding

from sapphire.classifier import _kmeans, _max_sim


def load_emotion_examples(path: str) -> dict[str, list[str]]:
    """Load positive/negative/high_arousal/low_arousal examples from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)

    def expand(items: list) -> list[str]:
        out = []
        for item in items:
            if isinstance(item, dict):
                for _ in range(item.get("weight", 1)):
                    out.append(item["text"])
            else:
                out.append(str(item))
        return out

    return {
        "positive": expand(data.get("positive", [])),
        "negative": expand(data.get("negative", [])),
        "high_arousal": expand(data.get("high_arousal", [])),
        "low_arousal": expand(data.get("low_arousal", [])),
    }


def compute_emotion_centroids(
    embedder: TextEmbedding,
    examples: dict[str, list[str]],
    k: int = 10,
) -> dict[str, np.ndarray]:
    """Compute k centroid vectors per pole via k-means."""
    centroids = {}
    for pole, texts in examples.items():
        emb = np.array(list(embedder.passage_embed(texts)))
        if len(emb) <= k:
            centroids[pole] = emb.mean(axis=0, keepdims=True)
        else:
            centroids[pole] = _kmeans(emb, k)
    return centroids


CENTROID_DIR = Path(__file__).resolve().parent.parent.parent / "centroids"


def save_emotion_centroids(
    centroids: dict[str, np.ndarray],
    path: str | Path = "",
) -> None:
    """Save emotion centroids to a .npz file."""
    dest = Path(path) if path else CENTROID_DIR / "emotion_centroids.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **centroids)


def load_emotion_centroids(
    path: str | Path = "",
) -> dict[str, np.ndarray] | None:
    """Load emotion centroids from a .npz file. Returns None if missing."""
    src = Path(path) if path else CENTROID_DIR / "emotion_centroids.npz"
    if not src.exists():
        return None
    data = np.load(src)
    return dict(data)


def emotion_centroid_path() -> str:
    return str(CENTROID_DIR / "emotion_centroids.npz")


def _axis_score(emb: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    """Max-signed cosine-similarity difference between two multicentroid poles."""
    norm = np.linalg.norm(emb)
    if norm == 0:
        return 0.0
    sim_pos = _max_sim(emb, norm, pos)
    sim_neg = _max_sim(emb, norm, neg)
    return sim_pos - sim_neg


def score_axes(
    emb: np.ndarray | None,
    centroids: dict[str, np.ndarray] | None,
) -> tuple[float, float]:
    """Return (valence, arousal), each in roughly [-1, 1]. (0, 0) if unavailable."""
    if emb is None or not centroids:
        return 0.0, 0.0
    valence = _axis_score(emb, centroids["positive"], centroids["negative"])
    arousal = _axis_score(emb, centroids["high_arousal"], centroids["low_arousal"])
    return valence, arousal


def get_default_emotion_examples_path() -> str:
    """Path to examples_emotion.yml -- walks up from src/sapphire/ to repo root."""
    return str(Path(__file__).resolve().parent.parent.parent / "examples_emotion.yml")


class EmotionState:
    """
    Per-conversation exponential moving average of (valence, arousal),
    decaying toward 0 on every update -- same idea as Jade's topic fatigue.

    NOTE: this decays toward the *recent average raw signal*, not literally
    toward 0 -- if incoming messages carry a persistent weak signal (e.g.
    casual affirmations reading as mildly positive), the state will settle
    near that value, not at 0. `deadzone` exists to zero out signal too weak
    to be meaningful, so genuinely flat conversation does pull the state
    back toward 0.
    """

    def __init__(self, decay: float = 0.85, deadzone: float = 0.06):
        self.decay = decay
        self.deadzone = deadzone
        self._state: dict[str, dict[str, float]] = {}

    def update(self, key: str, valence_delta: float, arousal_delta: float) -> dict[str, float]:
        if abs(valence_delta) < self.deadzone:
            valence_delta = 0.0
        if abs(arousal_delta) < self.deadzone:
            arousal_delta = 0.0

        s = self._state.setdefault(key, {"valence": 0.0, "arousal": 0.0})
        s["valence"] = s["valence"] * self.decay + valence_delta * (1 - self.decay)
        s["arousal"] = s["arousal"] * self.decay + arousal_delta * (1 - self.decay)
        return s

    def get(self, key: str) -> dict[str, float]:
        return self._state.get(key, {"valence": 0.0, "arousal": 0.0})
