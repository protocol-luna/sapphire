#!/usr/bin/env python3
"""Interactive 3D visualization of centroid classification space."""

import sys
import re
import numpy as np
import yaml
from pathlib import Path
from sklearn.decomposition import PCA
from fastembed import TextEmbedding
import plotly.graph_objects as go

CENTROID_DIR = Path(__file__).resolve().parent / ".." / "centroids"
EXAMPLES_PATH = Path(__file__).resolve().parent / ".." / "examples.yml"
COLORS = {"futile": "#4C72B0", "interessant": "#EDB232"}
CENTROID_COLORS = {"futile": "#1A3A6B", "interessant": "#B87D1A"}


def expand(items: list) -> list[str]:
    out = []
    for item in items:
        if isinstance(item, dict):
            for _ in range(item.get("weight", 1)):
                out.append(item["text"])
        else:
            out.append(str(item))
    return out


def main():
    test_sentence = None
    if len(sys.argv) > 1:
        test_sentence = " ".join(sys.argv[1:])

    print("Loading examples...")
    with open(EXAMPLES_PATH) as f:
        data = yaml.safe_load(f)
    futile_texts = expand(data.get("futile", []))
    inter_texts = expand(data.get("interessant", []))

    print(f"  {len(futile_texts)} futile, {len(inter_texts)} interessant")

    print("Embedding with bge-small-en-v1.5...")
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", max_length=128)
    all_texts = futile_texts + inter_texts
    labels = ["futile"] * len(futile_texts) + ["interessant"] * len(inter_texts)
    embeddings = np.array(list(embedder.passage_embed(all_texts)))
    print(f"  Embeddings shape: {embeddings.shape}")

    print("Loading centroids...")
    cent = np.load(CENTROID_DIR / "classifier_centroids.npz")
    f_cent = cent["futile"]
    i_cent = cent["interessant"]
    centroid_labels = ["centroid futile", "centroid interessant"]

    print("Computing 3D PCA...")
    all_pts = np.vstack([embeddings, f_cent.reshape(1, -1), i_cent.reshape(1, -1)])
    pca = PCA(n_components=3)
    proj = pca.fit_transform(all_pts)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    emb_proj = proj[:-2]
    cent_proj = proj[-2:]

    fig = go.Figure()

    for label, color in [("futile", COLORS["futile"]), ("interessant", COLORS["interessant"])]:
        mask = np.array([l == label for l in labels])
        pts = emb_proj[mask]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.5),
            name=label,
            showlegend=True,
        ))

    for label, color, name in [
        ("futile", CENTROID_COLORS["futile"], "Centroid Futile"),
        ("interessant", CENTROID_COLORS["interessant"], "Centroid Intéressant"),
    ]:
        idx = 0 if label == "futile" else 1
        fig.add_trace(go.Scatter3d(
            x=[cent_proj[idx, 0]], y=[cent_proj[idx, 1]], z=[cent_proj[idx, 2]],
            mode="markers",
            marker=dict(size=15, color=color, symbol="diamond"),
            name=name,
            showlegend=True,
        ))

    if test_sentence:
        print(f"Embedding test: \"{test_sentence}\"...")
        test_emb = np.array(list(embedder.passage_embed([test_sentence])))
        test_proj = pca.transform(test_emb)
        fig.add_trace(go.Scatter3d(
            x=[test_proj[0, 0]], y=[test_proj[0, 1]], z=[test_proj[0, 2]],
            mode="markers+text",
            marker=dict(size=12, color="#E74C3C", symbol="circle"),
            text=["Test"],
            textposition="top center",
            name=f'Test: "{test_sentence}"',
            showlegend=True,
        ))

    fig.update_layout(
        title="Classification par centroids : espace 3D (PCA)",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        legend=dict(x=0, y=1),
        width=1000,
        height=800,
        hovermode="closest",
    )

    out_path = Path(__file__).resolve().parent / "centroids_3d.html"
    fig.write_html(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
