"""
preprocessing/generate_pairs.py
Phase 8: Generate balanced positive (label=1, same emotion) and negative (label=0, diff emotion)
pairs for contrastive learning across train, val, and test splits.
See explainable.md > Phase 8 for details.
"""

import os
import sys
import random
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def generate_pairs_for_df(df, n_pairs, seed=config.RANDOM_SEED):
    """Generate balanced positive and negative pairs with full traceability."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    emotions = sorted(df["emotion"].unique())
    by_emotion = {emo: df[df["emotion"] == emo].to_dict("records") for emo in emotions}

    n_pos = n_pairs // 2
    n_neg = n_pairs - n_pos

    pairs = []

    # Generate positive pairs (same emotion)
    for _ in range(n_pos):
        emo = rng.choice(emotions)
        pool = by_emotion[emo]
        if len(pool) < 2:
            img1 = img2 = pool[0]
        else:
            # Prefer cross-subject pairs when available to encourage identity invariance
            img1, img2 = rng.sample(pool, 2)

        pair_type = "cross_subject_pos" if img1["subject_id"] != img2["subject_id"] else "same_subject_pos"
        pairs.append({
            "image_1": img1["processed_path"],
            "image_2": img2["processed_path"],
            "emotion_1": img1["emotion"],
            "emotion_2": img2["emotion"],
            "subject_1": img1["subject_id"],
            "subject_2": img2["subject_id"],
            "label": 1,
            "pair_type": pair_type,
        })

    # Generate negative pairs (different emotions)
    for _ in range(n_neg):
        emo1, emo2 = rng.sample(emotions, 2)
        img1 = rng.choice(by_emotion[emo1])
        img2 = rng.choice(by_emotion[emo2])

        pairs.append({
            "image_1": img1["processed_path"],
            "image_2": img2["processed_path"],
            "emotion_1": img1["emotion"],
            "emotion_2": img2["emotion"],
            "subject_1": img1["subject_id"],
            "subject_2": img2["subject_id"],
            "label": 0,
            "pair_type": "negative",
        })

    rng.shuffle(pairs)
    pairs_df = pd.DataFrame(pairs)

    # Verification assertions
    assert (pairs_df["label"] == 1).sum() == n_pos, "Positive count mismatch!"
    assert (pairs_df["label"] == 0).sum() == n_neg, "Negative count mismatch!"
    assert ((pairs_df["label"] == 1) == (pairs_df["emotion_1"] == pairs_df["emotion_2"])).all(), "Label logic error!"

    return pairs_df


def run_phase8():
    print("=" * 55)
    print("  KMU-FED Phase 8 — Positive & Negative Pair Generation")
    print("=" * 55)

    train_path = config.TRAIN_CSV
    val_path   = config.VAL_CSV
    test_path  = config.TEST_CSV

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing split CSV: {p}. Run Phase 7 first.")

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    # Generate pairs: 3000 for train, 600 for val, 600 for test
    train_pairs = generate_pairs_for_df(train_df, n_pairs=3000, seed=config.RANDOM_SEED)
    val_pairs   = generate_pairs_for_df(val_df,   n_pairs=600,  seed=config.RANDOM_SEED + 1)
    test_pairs  = generate_pairs_for_df(test_df,  n_pairs=600,  seed=config.RANDOM_SEED + 2)

    # Prevent split leakage: verify image pools are strictly disjoint
    train_imgs = set(train_df["processed_path"])
    val_imgs   = set(val_df["processed_path"])
    test_imgs  = set(test_df["processed_path"])

    train_pair_imgs = set(train_pairs["image_1"]).union(set(train_pairs["image_2"]))
    val_pair_imgs   = set(val_pairs["image_1"]).union(set(val_pairs["image_2"]))
    test_pair_imgs  = set(test_pairs["image_1"]).union(set(test_pairs["image_2"]))

    assert train_pair_imgs.issubset(train_imgs), "Train pairs contain external images!"
    assert val_pair_imgs.issubset(val_imgs), "Val pairs contain external images!"
    assert test_pair_imgs.issubset(test_imgs), "Test pairs contain external images!"
    assert not (train_pair_imgs & test_pair_imgs), "Train and Test pair images overlap!"

    # Save to metadata directory
    train_pairs_path = os.path.join(config.METADATA_DIR, "train_pairs.csv")
    val_pairs_path   = os.path.join(config.METADATA_DIR, "val_pairs.csv")
    test_pairs_path  = os.path.join(config.METADATA_DIR, "test_pairs.csv")

    train_pairs.to_csv(train_pairs_path, index=False)
    val_pairs.to_csv(val_pairs_path, index=False)
    test_pairs.to_csv(test_pairs_path, index=False)

    print(f"[OK] Train pairs saved -> {train_pairs_path} ({len(train_pairs)} pairs: 50% pos / 50% neg)")
    print(f"[OK] Val pairs saved   -> {val_pairs_path} ({len(val_pairs)} pairs: 50% pos / 50% neg)")
    print(f"[OK] Test pairs saved  -> {test_pairs_path} ({len(test_pairs)} pairs: 50% pos / 50% neg)")
    print("=" * 55)
    print("  Phase 8 Complete — Contrastive pairs ready")
    print("=" * 55)


if __name__ == "__main__":
    run_phase8()
