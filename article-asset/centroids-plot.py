#!/usr/bin/env python3
"""Generate interactive 3D centroid visualization for the article."""

import numpy as np
import yaml
from pathlib import Path
from sklearn.decomposition import PCA
from fastembed import TextEmbedding
import plotly.graph_objects as go

CENTROID_DIR = Path(__file__).resolve().parent.parent / "centroids"
EXAMPLES_PATH = Path(__file__).resolve().parent.parent / "examples.yml"
OUT_PATH = Path(__file__).resolve().parent / "centroids_plot_page.html"


def expand_with_texts(items):
    texts, display = [], []
    for item in items:
        if isinstance(item, dict):
            t = item["text"]
            for _ in range(item.get("weight", 1)):
                texts.append(t)
                display.append(t)
        else:
            texts.append(str(item))
            display.append(str(item))
    return texts, display


def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    with open(EXAMPLES_PATH) as f:
        data = yaml.safe_load(f)

    futile_texts, futile_display = expand_with_texts(data.get("futile", []))
    inter_texts, inter_display = expand_with_texts(data.get("interessant", []))
    all_texts = futile_texts + inter_texts
    all_display = futile_display + inter_display
    labels = ["futile"] * len(futile_texts) + ["interessant"] * len(inter_texts)

    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", max_length=128)
    embeddings = np.array(list(embedder.passage_embed(all_texts)))

    cent = np.load(CENTROID_DIR / "classifier_centroids.npz")
    f_cent, i_cent = cent["futile"], cent["interessant"]
    # f_cent and i_cent are (k, dim) -- flatten all centroids into PCA
    all_centroids = np.vstack([f_cent, i_cent])
    n_f = len(f_cent)
    all_pts = np.vstack([embeddings, all_centroids])
    pca = PCA(n_components=3)
    proj = pca.fit_transform(all_pts)
    emb_proj = proj[:-len(all_centroids)]
    cent_proj = proj[-len(all_centroids):]
    f_cent_proj = cent_proj[:n_f]
    i_cent_proj = cent_proj[n_f:]
    var = pca.explained_variance_ratio_.sum()

    hover_texts = []
    predicted = []
    mis_f_count = mis_i_count = 0
    for i in range(len(all_texts)):
        sim_f = float(np.max(f_cent @ embeddings[i] / (np.linalg.norm(embeddings[i]) * np.linalg.norm(f_cent, axis=1))))
        sim_i = float(np.max(i_cent @ embeddings[i] / (np.linalg.norm(embeddings[i]) * np.linalg.norm(i_cent, axis=1))))
        diff = sim_i - sim_f
        label = "INTERESTING" if diff > 0 else "FUTILE"
        predicted.append("interessant" if diff > 0 else "futile")
        if labels[i] == "futile" and predicted[-1] == "interessant":
            mis_f_count += 1
        elif labels[i] == "interessant" and predicted[-1] == "futile":
            mis_i_count += 1
        hover_texts.append(
            f"<b>{all_display[i]}</b><br>"
            f"Futile: {sim_f:.3f} | Interesting: {sim_i:.3f}<br>"
            f"Diff: {diff:+.3f} \u2192 {label}"
        )

    fig = go.Figure()

    for orig_label, base_color, group_name in [
        ("futile", "#4C72B0", "Futile"),
        ("interessant", "#EDB232", "Interesting"),
    ]:
        for is_mis, suffix, line_w, sz in [
            (False, "", 0, 3),
            (True, " (ambiguous)", 1, 4),
        ]:
            mask = np.array([
                labels[i] == orig_label and (predicted[i] != orig_label) == is_mis
                for i in range(len(labels))
            ])
            if not mask.any():
                continue
            pts = emb_proj[mask]
            txts = [hover_texts[i] for i in range(len(labels)) if mask[i]]
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(
                    size=sz, color=base_color,
                    opacity=0.8 if is_mis else 0.5,
                    line=dict(width=line_w, color="#333"),
                ),
                text=txts, hoverinfo="text",
                hoverlabel=dict(bgcolor=base_color if not is_mis else "#333"),
                name=f"{group_name}{suffix}",
            ))

    for idx in range(len(f_cent_proj)):
        color = "#1A3A6B" if idx == 0 else "#4A7AB5"
        fig.add_trace(go.Scatter3d(
            x=[f_cent_proj[idx, 0]], y=[f_cent_proj[idx, 1]], z=[f_cent_proj[idx, 2]],
            mode="markers",
            marker=dict(
                size=15 if idx == 0 else 9, color=color, symbol="diamond",
                line=dict(width=1, color="white"),
            ),
            name=f"Futile centroid {idx+1}" if n_f > 1 else "Futile centroid",
            hoverinfo="name",
        ))
    for idx in range(len(i_cent_proj)):
        color = "#B87D1A" if idx == 0 else "#D4A84B"
        fig.add_trace(go.Scatter3d(
            x=[i_cent_proj[idx, 0]], y=[i_cent_proj[idx, 1]], z=[i_cent_proj[idx, 2]],
            mode="markers",
            marker=dict(
                size=15 if idx == 0 else 9, color=color, symbol="diamond",
                line=dict(width=1, color="white"),
            ),
            name=f"Interesting centroid {idx+1}" if len(i_cent_proj) > 1 else "Interesting centroid",
            hoverinfo="name",
        ))

    test_sentences = ["lol", "i feel sad today"]
    test_embs = np.array(list(embedder.passage_embed(test_sentences)))
    test_proj = pca.transform(test_embs)

    for sent, proj_pt in zip(test_sentences, test_proj):
        emb = test_embs[test_sentences.index(sent)]
        sim_f = float(np.max(f_cent @ emb / (np.linalg.norm(emb) * np.linalg.norm(f_cent, axis=1))))
        sim_i = float(np.max(i_cent @ emb / (np.linalg.norm(emb) * np.linalg.norm(i_cent, axis=1))))
        diff = sim_i - sim_f
        label = "INTERESTING" if diff > 0 else "FUTILE"
        hover = (
            f"<b>{sent}</b><br>"
            f"Futile: {sim_f:.3f} | Interesting: {sim_i:.3f}<br>"
            f"Diff: {diff:+.3f} \u2192 {label}"
        )
        fig.add_trace(go.Scatter3d(
            x=[proj_pt[0]], y=[proj_pt[1]], z=[proj_pt[2]],
            mode="markers+text",
            marker=dict(
                size=10, color="#E74C3C", symbol="circle",
                line=dict(width=1, color="white"),
            ),
            text=[f'"{sent}"'],
            textposition="top center",
            textfont=dict(size=11, color="#E74C3C"),
            name="Example",
            hoverinfo="text", hovertext=[hover],
            hoverlabel=dict(bgcolor="#E74C3C", font=dict(color="white")),
        ))

    fig.update_layout(
        legend=dict(x=0, y=1, font=dict(size=11)),
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            bgcolor="#f8f9fa",
        ),
        paper_bgcolor="#f8f9fa",
        annotations=[dict(
            x=0, y=1.05, xref="paper", yref="paper",
            text=(
                f"{len(futile_texts)} futile ({mis_f_count} ambiguous) \u2022 "
                f"{len(inter_texts)} interesting ({mis_i_count} ambiguous) \u2022 "
                f"explained variance {var:.1%}"
            ),
            showarrow=False, font=dict(size=12, color="#666"),
        )],
    )

    html = fig.to_html(
        include_plotlyjs="cdn", full_html=True,
        div_id="centroid-plot-3d",
        default_width="100%", default_height="100%",
        config={"responsive": True, "displayModeBar": False},
    )
    OUT_PATH.write_text(html)
    print(f"Saved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
