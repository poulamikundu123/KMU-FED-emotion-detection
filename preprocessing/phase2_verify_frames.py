"""
preprocessing/phase2_verify_frames.py
Phase 2: Deep-verify images, organize frames into sequence subfolders under data/frames/,
and produce sequence-level metadata.
See explainable.md > Phase 2 for details.
"""

import os
import sys
import shutil
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def verify_image_integrity(filepath):
    """Deep check: header validity, decodability, dimensions, and color mode."""
    with Image.open(filepath) as img:
        img.verify()
    with Image.open(filepath) as img:
        img.load()  # decode pixel buffer to detect corruption
        return img.size, img.mode


def run_phase2():
    print("=" * 55)
    print("  KMU-FED Phase 2 — Frame Verification & Staging")
    print(f"  Source metadata : {config.METADATA_CSV}")
    print(f"  Target frames   : {config.FRAMES_DIR}")
    print("=" * 55)

    if not os.path.exists(config.METADATA_CSV):
        raise FileNotFoundError(f"Metadata CSV missing: {config.METADATA_CSV}. Run Phase 1 first.")

    df = pd.read_csv(config.METADATA_CSV)
    os.makedirs(config.FRAMES_DIR, exist_ok=True)

    verified_count = 0
    copied_count = 0
    errors = []

    # Preserve raw path column if not already present
    if "raw_image_path" not in df.columns:
        df["raw_image_path"] = df["image_path"]

    new_frame_paths = []

    for idx, row in df.iterrows():
        filename = row["filename"]
        seq_id = str(row["sequence_id"])

        # Determine source path (fallback to raw directory if moved)
        candidate_src = row["raw_image_path"]
        if not os.path.exists(candidate_src):
            candidate_src = os.path.join(config.RAW_DATA_DIR, filename)

        if not os.path.exists(candidate_src):
            errors.append(f"Missing file: {candidate_src}")
            new_frame_paths.append(row["image_path"])
            continue

        # Verify image integrity
        try:
            size, mode = verify_image_integrity(candidate_src)
            if mode != "RGB":
                errors.append(f"Unexpected mode {mode} for {filename}")
        except Exception as e:
            errors.append(f"Corrupt image {filename}: {e}")
            new_frame_paths.append(row["image_path"])
            continue

        verified_count += 1

        # Destination: data/frames/{sequence_id}/{filename}
        seq_dir = os.path.join(config.FRAMES_DIR, seq_id)
        os.makedirs(seq_dir, exist_ok=True)
        dest_path = os.path.join(seq_dir, filename)

        # Copy only if missing or size differs
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) != os.path.getsize(candidate_src):
            shutil.copy2(candidate_src, dest_path)
            copied_count += 1

        new_frame_paths.append(dest_path)

    if errors:
        print(f"[!] Warning: {len(errors)} verification issues encountered:")
        for err in errors[:5]:
            print(f"    - {err}")
        raise RuntimeError("Image verification failed. See errors above.")

    # Update metadata DataFrame with working frame paths
    df["image_path"] = new_frame_paths
    df.to_csv(config.METADATA_CSV, index=False)
    print(f"[OK] Updated metadata CSV with working frame paths ({len(df)} rows)")

    # Build sequence-level summary CSV
    seq_records = []
    grouped = df.groupby("sequence_id")
    for seq_id, group in grouped:
        frames_sorted = group.sort_values("frame_id")
        seq_records.append({
            "sequence_id": seq_id,
            "subject_id": group["subject_id"].iloc[0],
            "person_code": group["person_code"].iloc[0],
            "emotion_code": group["emotion_code"].iloc[0],
            "emotion": group["emotion"].iloc[0],
            "frame_count": len(group),
            "start_frame": frames_sorted["frame_id"].iloc[0],
            "end_frame": frames_sorted["frame_id"].iloc[-1],
            "sequence_folder": os.path.join(config.FRAMES_DIR, seq_id),
        })

    seq_df = pd.DataFrame(seq_records).sort_values("sequence_id")
    seq_csv_path = os.path.join(config.METADATA_DIR, "kmu_fed_sequences.csv")
    seq_df.to_csv(seq_csv_path, index=False)
    print(f"[OK] Generated sequence summary -> {seq_csv_path}")

    # Summary
    print("=" * 55)
    print("  Phase 2 Complete")
    print(f"  Verified images : {verified_count}/{len(df)}")
    print(f"  Sequences       : {len(seq_df)} subfolders created")
    print(f"  Frames staged   : {config.FRAMES_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    run_phase2()
