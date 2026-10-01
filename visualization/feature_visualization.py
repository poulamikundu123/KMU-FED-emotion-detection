"""
visualization/feature_visualization.py
Phase 17: 2D feature space projection and clustering visualization using t-SNE and PCA.
Generates publication-quality 2D scatter plots and computes clustering separation metrics.
See explainable.md > Phase 17 for details.
"""

import os
import sys
import json
import argparse
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Emotion class names and curated color palette
EMOTION_TO_IDX = {
    "Anger": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happiness": 3,
    "Sadness": 4,
    "Surprise": 5,
}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}
CLASS_NAMES = [IDX_TO_EMOTION[i] for i in range(len(EMOTION_TO_IDX))]

# Distinct color palette for 6 emotions
EMOTION_COLORS = {
    "Anger": "#E74C3C",      # Crimson Red
    "Disgust": "#27AE60",    # Emerald Green
    "Fear": "#8E44AD",       # Amethyst Purple
    "Happiness": "#F39C12",  # Amber Orange
    "Sadness": "#2980B9",    # Deep Blue
    "Surprise": "#16A085",   # Turquoise
}


def compute_tsne(features, perplexity=config.VIZ_TSNE_PERPLEXITY, n_iter=config.VIZ_TSNE_N_ITER, random_state=config.RANDOM_SEED):
    """Project high-dimensional feature vectors to 2D using t-SNE."""
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, len(features) // 4)),
        n_iter_without_progress=300,
        random_state=random_state,
        init="pca",
        learning_rate="auto"
    )
    return tsne.fit_transform(features)


def compute_pca(features, n_components=2, random_state=config.RANDOM_SEED):
    """Project high-dimensional feature vectors to 2D using PCA."""
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(features), pca.explained_variance_ratio_


def compute_clustering_metrics(features, labels):
    """Compute mathematical clustering separation metrics."""
    n_classes = len(np.unique(labels))
    if n_classes < 2 or len(features) <= n_classes:
        return {"silhouette_score": 0.0, "davies_bouldin_score": 0.0, "calinski_harabasz_score": 0.0}

    sil = float(silhouette_score(features, labels))
    db  = float(davies_bouldin_score(features, labels))
    ch  = float(calinski_harabasz_score(features, labels))
    return {
        "silhouette_score": round(sil, 4),
        "davies_bouldin_score": round(db, 4),
        "calinski_harabasz_score": round(ch, 2),
    }


def plot_scatter(embeddings_2d, labels, output_path, title, subtitle=None):
    """Render publication-grade 2D scatter plot colored by emotion class."""
    plt.figure(figsize=(9, 7.5))
    ax = plt.subplot(1, 1, 1)

    for i, c_name in enumerate(CLASS_NAMES):
        mask = (labels == i)
        if np.any(mask):
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=EMOTION_COLORS[c_name],
                label=f"{c_name} (n={mask.sum()})",
                alpha=0.85,
                edgecolors="white",
                linewidths=0.6,
                s=65
            )

    plt.title(title, fontsize=14, pad=14, fontweight="bold")
    if subtitle:
        plt.suptitle(subtitle, fontsize=10, y=0.92, color="#555555")

    plt.xlabel("Component 1", fontsize=11, labelpad=8)
    plt.ylabel("Component 2", fontsize=11, labelpad=8)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_phase17(output_dir=None):
    print("=" * 60)
    print("  KMU-FED Phase 17 — Feature Space Visualization")
    print("=" * 60)

    if output_dir is None:
        output_dir = config.RESULTS_KMU_FED
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load extracted feature arrays
    tr_f = np.load(os.path.join(config.FEATURES_DIR, "train_features.npy"))
    tr_l = np.load(os.path.join(config.FEATURES_DIR, "train_labels.npy"))
    te_f = np.load(os.path.join(config.FEATURES_DIR, "test_features.npy"))
    te_l = np.load(os.path.join(config.FEATURES_DIR, "test_labels.npy"))
    va_f = np.load(os.path.join(config.FEATURES_DIR, "val_features.npy"))
    va_l = np.load(os.path.join(config.FEATURES_DIR, "val_labels.npy"))

    all_f = np.concatenate([tr_f, va_f, te_f], axis=0)
    all_l = np.concatenate([tr_l, va_l, te_l], axis=0)

    print(f"[OK] Loaded features: Train {tr_f.shape} | Val {va_f.shape} | Test {te_f.shape} | Combined {all_f.shape}")

    # 2. Compute clustering separation metrics
    metrics = {
        "test_set": compute_clustering_metrics(te_f, te_l),
        "train_set": compute_clustering_metrics(tr_f, tr_l),
        "all_data": compute_clustering_metrics(all_f, all_l),
    }

    metrics_path = os.path.join(output_dir, "clustering_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[OK] Saved clustering separation metrics to {metrics_path}")

    # 3. 2D t-SNE on Test Set Features
    print("  Computing t-SNE projection on Test Set (156 samples)...")
    te_tsne = compute_tsne(te_f, perplexity=25)
    te_tsne_path = os.path.join(output_dir, "tsne_features_test.png")
    plot_scatter(
        te_tsne, te_l, te_tsne_path,
        title="t-SNE Projection of 1280-dim Features (Test Set)",
        subtitle=f"Unseen Subjects 2 & 11 | Silhouette: {metrics['test_set']['silhouette_score']:.3f}"
    )
    print(f"[OK] Saved {te_tsne_path}")

    # 4. 2D PCA on Test Set Features
    print("  Computing PCA projection on Test Set...")
    te_pca, var_ratio = compute_pca(te_f, n_components=2)
    te_pca_path = os.path.join(output_dir, "pca_features_test.png")
    plot_scatter(
        te_pca, te_l, te_pca_path,
        title="PCA Projection of 1280-dim Features (Test Set)",
        subtitle=f"Explained Variance: PC1={var_ratio[0]*100:.1f}%, PC2={var_ratio[1]*100:.1f}%"
    )
    print(f"[OK] Saved {te_pca_path}")

    # 5. 2D t-SNE on Training Set Features
    print("  Computing t-SNE projection on Training Set (732 samples)...")
    tr_tsne = compute_tsne(tr_f, perplexity=30)
    tr_tsne_path = os.path.join(output_dir, "tsne_features_train.png")
    plot_scatter(
        tr_tsne, tr_l, tr_tsne_path,
        title="t-SNE Projection of 1280-dim Features (Training Set)",
        subtitle=f"8 Training Subjects | Silhouette: {metrics['train_set']['silhouette_score']:.3f}"
    )
    print(f"[OK] Saved {tr_tsne_path}")

    # 6. 2D t-SNE on Combined Dataset
    print("  Computing t-SNE projection on All 1,045 Samples...")
    all_tsne = compute_tsne(all_f, perplexity=30)
    all_tsne_path = os.path.join(output_dir, "tsne_features_all.png")
    plot_scatter(
        all_tsne, all_l, all_tsne_path,
        title="t-SNE Projection of 1280-dim Features (Entire KMU-FED Dataset)",
        subtitle=f"All 12 Subjects (1,045 faces) | Silhouette: {metrics['all_data']['silhouette_score']:.3f}"
    )
    print(f"[OK] Saved {all_tsne_path}")

    # 7. Print Console Summary
    print("\n" + "=" * 60)
    print("  KMU-FED Feature Space Clustering Summary")
    print("=" * 60)
    print(f"  {'Split':<12} | {'Silhouette':<12} | {'Davies-Bouldin':<15} | {'Calinski-Harabasz':<18}")
    print("  " + "-" * 57)
    for s_key in ["train_set", "test_set", "all_data"]:
        m = metrics[s_key]
        print(f"  {s_key:<12} | {m['silhouette_score']:<12.4f} | {m['davies_bouldin_score']:<15.4f} | {m['calinski_harabasz_score']:<18.2f}")
    print("=" * 60)
    print(f"  Generated Visualizations in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KMU-FED Phase 17 Feature Visualization")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()

    run_phase17(output_dir=args.output_dir)
