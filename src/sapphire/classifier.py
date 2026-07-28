"""
Sapphire classifier -- multicentroid k-means similarity.

Computes cosine similarity between a query and k centroids per class (futile /
interessant). The class score is the maximum similarity across its k centroids.
"""

from pathlib import Path

import numpy as np
import yaml
from fastembed import TextEmbedding


def load_examples(path: str) -> tuple[list[str], list[str]]:
    """Load futile and interessant examples from a YAML file."""
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

    return expand(data.get("futile", [])), expand(data.get("interessant", []))


def _kmeans(data: np.ndarray, k: int, max_iters: int = 20) -> np.ndarray:
    """Simple k-means clustering. Returns (k, dim) centroids."""
    n, _dim = data.shape
    k = min(k, n)
    rng = np.random.default_rng(42)
    idx = rng.choice(n, k, replace=False)
    centroids = data[idx].copy()
    for _ in range(max_iters):
        dists = np.linalg.norm(data[:, None] - centroids[None], axis=2)
        labels = np.argmin(dists, axis=1)
        new = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        new = np.where(np.isnan(new), centroids, new)
        if np.allclose(centroids, new):
            break
        centroids = new
    return centroids


def compute_centroids(
    embedder: TextEmbedding,
    futile: list[str],
    interessant: list[str],
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute k centroid vectors per class via k-means."""
    f_emb = np.array(list(embedder.passage_embed(futile)))
    i_emb = np.array(list(embedder.passage_embed(interessant)))
    if len(f_emb) <= k:
        f_cent = f_emb.mean(axis=0, keepdims=True)
    else:
        f_cent = _kmeans(f_emb, k)
    if len(i_emb) <= k:
        i_cent = i_emb.mean(axis=0, keepdims=True)
    else:
        i_cent = _kmeans(i_emb, k)
    return f_cent, i_cent


CENTROID_DIR = Path(__file__).resolve().parent.parent.parent / "centroids"


def save_centroids(
    futile_centroid: np.ndarray,
    interessant_centroid: np.ndarray,
    path: str | Path = "",
) -> None:
    """Save classification centroids to a .npz file."""
    dest = Path(path) if path else CENTROID_DIR / "classifier_centroids.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, futile=futile_centroid, interessant=interessant_centroid)


def load_centroids(
    path: str | Path = "",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load classification centroids from a .npz file. Returns (None, None) if missing."""
    src = Path(path) if path else CENTROID_DIR / "classifier_centroids.npz"
    if not src.exists():
        return None, None
    data = np.load(src)
    return data["futile"], data["interessant"]


def centroid_path() -> str:
    return str(CENTROID_DIR / "classifier_centroids.npz")


def _max_sim(emb: np.ndarray, norm: float, centroids: np.ndarray) -> float:
    """Maximum cosine similarity between `emb` and any centroid row."""
    dots = centroids @ emb
    c_norms = np.linalg.norm(centroids, axis=1)
    sims = dots / (norm * c_norms)
    return float(np.max(sims))


def classify(
    text: str,
    embedder: TextEmbedding | None,
    futile_centroid: np.ndarray | None,
    interessant_centroid: np.ndarray | None,
    precomputed_emb: np.ndarray | None = None,
) -> tuple[str, float, float, float]:
    """Classify text as FUTILE or INTERESSANT via max cosine similarity over k centroids.

    Each centroid array is (k, dim). The per-class score is the max similarity
    across all k centroids for that class.

    Pass `precomputed_emb` to skip the embedder call when the caller already
    computed the embedding for this text (e.g. to also feed emotion.score_axes
    without embedding the same text twice).
    """
    if embedder is None or futile_centroid is None or interessant_centroid is None:
        return "FUTILE", 0.0, 0.0, 0.0
    emb = (
        precomputed_emb
        if precomputed_emb is not None
        else next(embedder.query_embed(text))
    )
    norm = np.linalg.norm(emb)
    if norm == 0:
        return "FUTILE", 0.0, 0.0, 0.0

    sim_f = _max_sim(emb, norm, futile_centroid)
    sim_i = _max_sim(emb, norm, interessant_centroid)
    diff = sim_i - sim_f
    label = "INTERESSANT" if diff > 0 else "FUTILE"
    return label, abs(diff), sim_f, sim_i


def get_default_examples_path() -> str:
    """Path to examples.yml -- walks up from src/sapphire/ to repo root."""
    return str(Path(__file__).resolve().parent.parent.parent / "examples.yml")
