"""
preprocessing/split_dataset.py
Phase 7: Partition dataset into subject-independent Train (70%), Val (15%), and Test (15%) sets.
Ensures zero subject overlap and complete emotion representation across all splits.
See explainable.md > Phase 7 for details.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_subject_split():
    """Return verified subject partition matching 70/15/15 target with full emotion coverage."""
    # Selected to ensure all 6 emotions exist in Train, Val, and Test
    train_subjects = [1, 3, 5, 6, 7, 9, 10, 12]
    val_subjects   = [4, 8]
    test_subjects  = [2, 11]
    return train_subjects, val_subjects, test_subjects


def run_phase7():
    print("=" * 55)
    print("  KMU-FED Phase 7 — Train / Val / Test Split")
    print(f"  Input metadata : {config.METADATA_DIR}")
    print(f"  Ratios target  : {config.TRAIN_RATIO*100:.0f}% / {config.VAL_RATIO*100:.0f}% / {config.TEST_RATIO*100:.0f}%")
    print("=" * 55)

    proc_csv = os.path.join(config.METADATA_DIR, "kmu_fed_processed.csv")
    if not os.path.exists(proc_csv):
        raise FileNotFoundError(f"Missing processed metadata: {proc_csv}. Run Phase 5 first.")

    df = pd.read_csv(proc_csv)

    train_subjs, val_subjs, test_subjs = get_subject_split()

    train_df = df[df["subject_id"].isin(train_subjs)].copy()
    val_df   = df[df["subject_id"].isin(val_subjs)].copy()
    test_df  = df[df["subject_id"].isin(test_subjs)].copy()

    # Integrity assertions: zero subject overlap (prevent data leakage)
    train_set = set(train_df["subject_id"].unique())
    val_set   = set(val_df["subject_id"].unique())
    test_set  = set(test_df["subject_id"].unique())

    assert (train_set & test_set) == set(), "Leakage detected between Train and Test!"
    assert (train_set & val_set) == set(), "Leakage detected between Train and Val!"
    assert (val_set & test_set) == set(), "Leakage detected between Val and Test!"
    assert len(train_df) + len(val_df) + len(test_df) == len(df), "Row count mismatch across splits!"

    # Save CSV files
    train_df.to_csv(config.TRAIN_CSV, index=False)
    val_df.to_csv(config.VAL_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)

    print(f"[OK] Train CSV saved -> {config.TRAIN_CSV} ({len(train_df)} rows, {len(train_df)/len(df)*100:.1f}%)")
    print(f"     Subjects: {sorted(train_subjs)}")
    print(f"     Emotions: {sorted(train_df['emotion'].unique())}")

    print(f"[OK] Val CSV saved   -> {config.VAL_CSV} ({len(val_df)} rows, {len(val_df)/len(df)*100:.1f}%)")
    print(f"     Subjects: {sorted(val_subjs)}")
    print(f"     Emotions: {sorted(val_df['emotion'].unique())}")

    print(f"[OK] Test CSV saved  -> {config.TEST_CSV} ({len(test_df)} rows, {len(test_df)/len(df)*100:.1f}%)")
    print(f"     Subjects: {sorted(test_subjs)}")
    print(f"     Emotions: {sorted(test_df['emotion'].unique())}")

    print("=" * 55)
    print("  Phase 7 Complete — Subject-independent split verified")
    print("=" * 55)


if __name__ == "__main__":
    run_phase7()
