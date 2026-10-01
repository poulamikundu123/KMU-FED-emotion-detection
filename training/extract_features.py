"""
training/extract_features.py
Phase 13: Extract and cache 1280-dimensional feature vectors from the trained
EfficientNet encoder across train, val, and test splits.
See explainable.md > Phase 13 for details.
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch  # type: ignore


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.efficientnet_encoder import EfficientNetEncoder
from preprocessing.preprocessing import get_default_transform


# Mapping emotion names to continuous integer class indices (0 to 5)
EMOTION_TO_IDX = {
    "Anger": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happiness": 3,
    "Sadness": 4,
    "Surprise": 5,
}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}


def extract_split_features(encoder, df, transform, device, batch_size=32):
    """Extract 1280-dim feature vectors and labels for a split DataFrame."""
    feature_list = []
    label_list = []
    meta_records = []

    total = len(df)
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_df = df.iloc[start_idx:end_idx]

        tensors = []
        for _, row in batch_df.iterrows():
            img_path = row["processed_path"]
            img = Image.open(img_path).convert("RGB")
            tensors.append(transform(img))

            emo_name = row["emotion"]
            emo_idx = EMOTION_TO_IDX[emo_name]
            label_list.append(emo_idx)
            meta_records.append({
                "sequence_id": row["sequence_id"],
                "filename": row["filename"],
                "subject_id": row["subject_id"],
                "emotion": emo_name,
                "emotion_idx": emo_idx,
            })

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = encoder(batch_tensor)  # (batch_size, 1280)

        feature_list.append(feats.cpu().numpy())

    features = np.concatenate(feature_list, axis=0)
    labels = np.array(label_list, dtype=np.int64)
    meta_df = pd.DataFrame(meta_records)

    return features, labels, meta_df


def run_phase13(encoder_weights=None, batch_size=32):
    print("=" * 55)
    print("  KMU-FED Phase 13 — Feature Representation Extraction")
    print(f"  Target features dir : {config.FEATURES_DIR}")
    print(f"  Feature dimension   : 1280")
    print("=" * 55)

    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # Determine encoder checkpoint path
    if encoder_weights is None:
        encoder_weights = os.path.join(config.CHECKPOINTS_DIR, "contrastive_encoder.pth")

    if not os.path.exists(encoder_weights):
        # Fallback to best_contrastive_model if standalone encoder weights not found
        best_siamese = os.path.join(config.CHECKPOINTS_DIR, "best_contrastive_model.pth")
        if os.path.exists(best_siamese):
            print(f"[Info] Extracting encoder from {best_siamese}")
            ckpt = torch.load(best_siamese, map_location=device, weights_only=False)
            encoder = EfficientNetEncoder(pretrained=False)
            # Filter state_dict keys with prefix 'encoder.'
            enc_state = {k.replace("encoder.", ""): v for k, v in ckpt["model_state_dict"].items() if k.startswith("encoder.")}
            encoder.load_state_dict(enc_state)
        else:
            raise FileNotFoundError(f"Missing encoder weights: {encoder_weights}. Run Phase 12 first.")
    else:
        print(f"[OK] Loading trained encoder weights: {encoder_weights}")
        encoder = EfficientNetEncoder(pretrained=False)
        encoder.load_state_dict(torch.load(encoder_weights, map_location=device, weights_only=False))

    encoder.to(device)
    encoder.eval()

    transform = get_default_transform()

    splits = {
        "train": config.TRAIN_CSV,
        "val": config.VAL_CSV,
        "test": config.TEST_CSV,
    }

    summary = {}
    for split_name, csv_path in splits.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing split CSV: {csv_path}. Run Phase 7 first.")

        df = pd.read_csv(csv_path)
        print(f"  Extracting {split_name} features ({len(df)} images)...")

        feats, labels, meta_df = extract_split_features(encoder, df, transform, device, batch_size=batch_size)

        # Save numpy arrays and metadata
        feat_path = os.path.join(config.FEATURES_DIR, f"{split_name}_features.npy")
        lbl_path  = os.path.join(config.FEATURES_DIR, f"{split_name}_labels.npy")
        meta_path = os.path.join(config.FEATURES_DIR, f"{split_name}_metadata.csv")

        np.save(feat_path, feats)
        np.save(lbl_path, labels)
        meta_df.to_csv(meta_path, index=False)

        summary[split_name] = {
            "features_shape": feats.shape,
            "labels_shape": labels.shape,
            "classes": len(np.unique(labels)),
        }
        print(f"  [OK] Saved {split_name}_features.npy {feats.shape} & labels {labels.shape}")

    print("\n" + "=" * 55)
    print("  Phase 13 Complete — All features cached to disk")
    for s_name, info in summary.items():
        print(f"    {s_name:<6} : Features {info['features_shape']} | Labels {info['labels_shape']} | Classes {info['classes']}")
    print(f"  Output folder : {config.FEATURES_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    run_phase13()
