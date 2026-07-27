#!/usr/bin/env python3
"""Export centroid data as JSON + generate an article-ready HTML snippet."""

import json
import numpy as np
import yaml
from pathlib import Path
from sklearn.decomposition import PCA
from fastembed import TextEmbedding

CENTROID_DIR = Path(__file__).resolve().parent / "centroids"
EXAMPLES_PATH = Path(__file__).resolve().parent / "examples.yml"


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

    print("Loading centroids...")
    cent = np.load(CENTROID_DIR / "classifier_centroids.npz")
    f_cent = cent["futile"]
    i_cent = cent["interessant"]

    print("Computing 3D PCA...")
    all_pts = np.vstack([embeddings, f_cent.reshape(1, -1), i_cent.reshape(1, -1)])
    pca = PCA(n_components=3)
    proj = pca.fit_transform(all_pts)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    import plotly.graph_objects as go

    emb_proj = proj[:-2]
    cent_proj = proj[-2:]

    fig = go.Figure()

    # Futile points (blue)
    mask_f = np.array([l == "futile" for l in labels])
    pts_f = emb_proj[mask_f]
    fig.add_trace(go.Scatter3d(
        x=pts_f[:, 0], y=pts_f[:, 1], z=pts_f[:, 2],
        mode="markers",
        marker=dict(size=3, color="#4C72B0", opacity=0.5),
        name="Futile",
    ))

    # Interessant points (yellow)
    mask_i = np.array([l == "interessant" for l in labels])
    pts_i = emb_proj[mask_i]
    fig.add_trace(go.Scatter3d(
        x=pts_i[:, 0], y=pts_i[:, 1], z=pts_i[:, 2],
        mode="markers",
        marker=dict(size=3, color="#EDB232", opacity=0.5),
        name="Intéressant",
    ))

    # Centroids
    for idx, (label, color, name) in enumerate([
        ("futile", "#1A3A6B", "Centroïde Futile"),
        ("interessant", "#B87D1A", "Centroïde Intéressant"),
    ]):
        fig.add_trace(go.Scatter3d(
            x=[cent_proj[idx, 0]], y=[cent_proj[idx, 1]], z=[cent_proj[idx, 2]],
            mode="markers",
            marker=dict(size=15, color=color, symbol="diamond"),
            name=name,
        ))

    fig.update_layout(
        title="Classification par centroïdes : projection PCA 3D",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        legend=dict(x=0, y=1),
        width=900,
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Export as div+JSON (plotly.js loaded from CDN)
    html = fig.to_html(include_plotlyjs="cdn", full_html=False,
                       div_id="centroid-plot-3d",
                       default_width="100%", default_height="500px")
    out_path = Path(__file__).resolve().parent / "centroids_3d_embed.html"
    out_path.write_text(html)
    print(f"Saved embeddable HTML: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
