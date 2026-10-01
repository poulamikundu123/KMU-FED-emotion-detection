"""
evaluation/evaluate.py
Phase 16: Comprehensive evaluation of the trained emotion recognition model on the held-out test set.
Computes Accuracy, Precision, Recall, F1-scores, Confusion Matrices, and saves visualization plots.
See explainable.md > Phase 16 for details.
"""

import os
import sys
import json
import argparse
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import torch  # type: ignore

from sklearn.metrics import (  # type: ignore
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.emotion_classifier import EmotionClassifier


# Continuous emotion class mapping
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


def plot_confusion_matrix(cm, class_names, output_path, title="Confusion Matrix", normalize=False):
    """Plot and save seaborn heatmap of confusion matrix."""
    plt.figure(figsize=(8, 6.5))
    if normalize:
        # Normalize by true class row
        cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
        sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, cbar=True)
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, cbar=True)

    plt.title(title, fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Predicted Emotion", fontsize=12, labelpad=8)
    plt.ylabel("True Emotion", fontsize=12, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_per_class_metrics(report_df, class_names, output_path):
    """Plot grouped bar chart of Precision, Recall, and F1 per class."""
    sub_df = report_df.loc[class_names, ["precision", "recall", "f1-score"]] * 100.0

    ax = sub_df.plot(kind="bar", figsize=(10, 6), width=0.75, colormap="viridis")
    plt.title("Per-Class Emotion Recognition Metrics (Test Set)", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Emotion Class", fontsize=12, labelpad=8)
    plt.ylabel("Score (%)", fontsize=12, labelpad=8)
    plt.ylim(0, 110)
    plt.xticks(rotation=0, fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(["Precision", "Recall", "F1-Score"], loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_phase16(checkpoint_path=None, output_dir=None, save_plots=True):
    print("=" * 60)
    print("  KMU-FED Phase 16 — Comprehensive Model Evaluation")
    print("=" * 60)

    if output_dir is None:
        output_dir = config.RESULTS_KMU_FED
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # 1. Load test features and labels
    te_feat_path = os.path.join(config.FEATURES_DIR, "test_features.npy")
    te_lbl_path  = os.path.join(config.FEATURES_DIR, "test_labels.npy")
    te_meta_path = os.path.join(config.FEATURES_DIR, "test_metadata.csv")

    if not os.path.exists(te_feat_path) or not os.path.exists(te_lbl_path):
        raise FileNotFoundError(f"Missing test features in {config.FEATURES_DIR}. Run Phase 13 first.")

    test_features = np.load(te_feat_path)
    test_labels   = np.load(te_lbl_path)

    if os.path.exists(te_meta_path):
        meta_df = pd.read_csv(te_meta_path)
    elif os.path.exists(config.TEST_CSV):
        meta_df = pd.read_csv(config.TEST_CSV)
    else:
        meta_df = pd.DataFrame({"sample_idx": range(len(test_labels))})

    # 2. Determine and load checkpoint
    if checkpoint_path is None:
        stage_a_ckpt = os.path.join(config.CHECKPOINTS_DIR, "best_classifier_stageA.pth")
        e2e_ckpt = os.path.join(config.CHECKPOINTS_DIR, "best_emotion_model.pth")
        if os.path.exists(stage_a_ckpt):
            checkpoint_path = stage_a_ckpt
        elif os.path.exists(e2e_ckpt):
            checkpoint_path = e2e_ckpt
        else:
            raise FileNotFoundError(f"No checkpoint found in {config.CHECKPOINTS_DIR}. Run Phase 15 first.")

    print(f"[OK] Loading evaluation checkpoint: {checkpoint_path}")
    classifier = EmotionClassifier(feature_dim=1280, num_classes=len(CLASS_NAMES))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle state_dict if coming from full EndToEndEmotionModel
    if any(k.startswith("classifier.") for k in ckpt.keys()):
        cls_state = {k.replace("classifier.", ""): v for k, v in ckpt.items() if k.startswith("classifier.")}
        classifier.load_state_dict(cls_state)
    else:
        classifier.load_state_dict(ckpt)

    classifier.to(device)
    classifier.eval()

    # 3. Model Inference on Test Set
    feats_tensor = torch.from_numpy(test_features).float().to(device)
    with torch.no_grad():
        logits = classifier(feats_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    # 4. Compute Aggregate Metrics
    acc = accuracy_score(test_labels, preds)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(test_labels, preds, average="macro", zero_division=0)
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(test_labels, preds, average="weighted", zero_division=0)

    # 5. Classification Report DataFrame
    report_dict = classification_report(test_labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = os.path.join(output_dir, "classification_report.csv")
    report_df.to_csv(report_csv_path)
    print(f"[OK] Saved classification report to {report_csv_path}")

    # 6. Confusion Matrix
    cm = confusion_matrix(test_labels, preds, labels=range(len(CLASS_NAMES)))
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in CLASS_NAMES], columns=[f"Pred_{c}" for c in CLASS_NAMES])
    cm_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)
    print(f"[OK] Saved confusion matrix values to {cm_csv_path}")

    # 7. Per-Sample Detailed Predictions
    preds_df = meta_df.copy()
    preds_df["true_emotion_idx"] = test_labels
    preds_df["true_emotion"] = [IDX_TO_EMOTION[i] for i in test_labels]
    preds_df["predicted_emotion_idx"] = preds
    preds_df["predicted_emotion"] = [IDX_TO_EMOTION[i] for i in preds]
    preds_df["correct"] = (preds == test_labels)
    preds_df["confidence"] = probs.max(axis=1)

    # Probabilities per class
    for i, name in enumerate(CLASS_NAMES):
        preds_df[f"prob_{name}"] = probs[:, i]

    preds_csv_path = os.path.join(output_dir, "per_sample_predictions.csv")
    preds_df.to_csv(preds_csv_path, index=False)
    print(f"[OK] Saved per-sample predictions to {preds_csv_path}")

    # 8. Evaluation Summary JSON
    summary = {
        "dataset": "KMU-FED",
        "split": "test",
        "total_test_samples": int(len(test_labels)),
        "accuracy": round(float(acc) * 100.0, 2),
        "macro_precision": round(float(macro_prec) * 100.0, 2),
        "macro_recall": round(float(macro_rec) * 100.0, 2),
        "macro_f1": round(float(macro_f1) * 100.0, 2),
        "weighted_precision": round(float(weighted_prec) * 100.0, 2),
        "weighted_recall": round(float(weighted_rec) * 100.0, 2),
        "weighted_f1": round(float(weighted_f1) * 100.0, 2),
        "per_class_accuracy": {
            name: round(float((preds[test_labels == i] == i).sum() / max((test_labels == i).sum(), 1)) * 100.0, 2)
            for i, name in enumerate(CLASS_NAMES)
        }
    }
    summary_json_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[OK] Saved evaluation summary JSON to {summary_json_path}")

    # 9. Visual Plots
    if save_plots:
        cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
        cm_norm_plot_path = os.path.join(output_dir, "confusion_matrix_normalized.png")
        bar_plot_path = os.path.join(output_dir, "per_class_metrics.png")

        plot_confusion_matrix(cm, CLASS_NAMES, cm_plot_path, title="KMU-FED Confusion Matrix (Counts)", normalize=False)
        plot_confusion_matrix(cm, CLASS_NAMES, cm_norm_plot_path, title="KMU-FED Confusion Matrix (Normalized %)", normalize=True)
        plot_per_class_metrics(report_df, CLASS_NAMES, bar_plot_path)

        print(f"[OK] Saved visual plots:\n     - {cm_plot_path}\n     - {cm_norm_plot_path}\n     - {bar_plot_path}")

    # 10. Terminal Formatted Output
    print("\n" + "=" * 60)
    print(f"  KMU-FED Test Set Evaluation Summary ({len(test_labels)} samples)")
    print("=" * 60)
    print(f"  Overall Accuracy    : {summary['accuracy']}%")
    print(f"  Macro Precision     : {summary['macro_precision']}%")
    print(f"  Macro Recall        : {summary['macro_recall']}%")
    print(f"  Macro F1-Score      : {summary['macro_f1']}%")
    print(f"  Weighted F1-Score   : {summary['weighted_f1']}%")
    print("-" * 60)
    print("  Per-Class Metrics:")
    print(f"  {'Emotion':<11} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("  " + "-" * 56)
    for c in CLASS_NAMES:
        row = report_df.loc[c]
        print(f"  {c:<11} | {row['precision']*100:8.2f}% | {row['recall']*100:8.2f}% | {row['f1-score']*100:8.2f}% | {int(row['support']):<8}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KMU-FED Phase 16 Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output results directory")
    parser.add_argument("--no_plots", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    run_phase16(checkpoint_path=args.checkpoint, output_dir=args.output_dir, save_plots=not args.no_plots)
