"""
Sapphire classifier -- embedding centroid similarity.

Computes cosine similarity between a query and two pre-computed centroids
(futile / interessant) derived from curated examples.
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


def compute_centroids(
    embedder: TextEmbedding,
    futile: list[str],
    interessant: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute centroid vectors by averaging all example embeddings."""
    f_emb = np.array(list(embedder.passage_embed(futile)))
    i_emb = np.array(list(embedder.passage_embed(interessant)))
    return f_emb.mean(axis=0), i_emb.mean(axis=0)


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


def classify(
    text: str,
    embedder: TextEmbedding | None,
    futile_centroid: np.ndarray | None,
    interessant_centroid: np.ndarray | None,
    precomputed_emb: np.ndarray | None = None,
) -> tuple[str, float, float, float]:
    """Classify text as FUTILE or INTERESSANT via cosine similarity.

    Pass `precomputed_emb` to skip the embedder call when the caller already
    computed the embedding for this text (e.g. to also feed emotion.score_axes
    without embedding the same text twice).
    """
    if embedder is None or futile_centroid is None or interessant_centroid is None:
        return "FUTILE", 0.0, 0.0, 0.0
    emb = precomputed_emb if precomputed_emb is not None else next(embedder.query_embed(text))
    norm = np.linalg.norm(emb)
    if norm == 0:
        return "FUTILE", 0.0, 0.0, 0.0
    sim_f = float(np.dot(emb, futile_centroid) / (norm * np.linalg.norm(futile_centroid)))
    sim_i = float(np.dot(emb, interessant_centroid) / (norm * np.linalg.norm(interessant_centroid)))
    diff = sim_i - sim_f
    label = "INTERESSANT" if diff > 0 else "FUTILE"
    return label, abs(diff), sim_f, sim_i


def get_default_examples_path() -> str:
    """Path to examples.yml -- walks up from src/sapphire/ to repo root."""
    return str(Path(__file__).resolve().parent.parent.parent / "examples.yml")
