"""
preprocessing/phase1_inspect_dataset.py
Phase 1: Scan the raw KMU-FED folder, parse filenames, exclude duplicates,
validate images, and write kmu_fed_metadata.csv + dataset_statistics_report.txt
See explainable.md > Phase 1 for a full walkthrough.
"""

import os
import sys
import csv
import argparse
import hashlib
from datetime import datetime
from collections import defaultdict

# Add project root to path so config.py can be imported from any directory
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import config


def parse_args():
    """Allow overriding default paths from the command line."""
    parser = argparse.ArgumentParser(
        description="Phase 1: KMU-FED Dataset Inspection"
    )
    parser.add_argument("--raw_dir",    default=config.RAW_DATA_DIR,  help="Path to raw KMU-FED folder")
    parser.add_argument("--output_dir", default=config.METADATA_DIR,  help="Where to save metadata outputs")
    return parser.parse_args()


def parse_filename(filename):
    """
    Parse a KMU-FED filename into its 4 parts.
    Pattern: {SubjectID}_{EmotionCode}_{PersonCode}_{FrameNumber}.jpg
    Returns a dict, or None if the filename doesn't match.
    """
    name, ext = os.path.splitext(filename)
    if ext.lower() != ".jpg":
        return None

    parts = name.split("_")
    if len(parts) != 4:
        return None

    subject_id, emotion_code, person_code, frame_id = parts

    if emotion_code not in config.EMOTION_CODE_MAP:
        return None

    return {
        "subject_id":   subject_id,
        "emotion_code": emotion_code,
        "person_code":  person_code,
        "frame_id":     frame_id,
        "emotion_name": config.EMOTION_CODE_MAP[emotion_code],
    }


def is_known_duplicate(filename):
    """Return True for files that are confirmed byte-for-byte copies (Kaggle artifact)."""
    return filename.startswith(config.DUPLICATE_PREFIX_EXCLUDE)


def compute_md5(filepath, chunk_size=8192):
    """Compute MD5 hash of a file for integrity verification."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def inspect_dataset(raw_dir, output_dir):
    """
    Main inspection loop: scan every file, parse, validate, and collect records.
    Returns (valid_records, stats_dict).
    """
    print(f"\n{'='*55}")
    print("  KMU-FED Phase 1 — Dataset Inspection")
    print(f"  Raw dir : {raw_dir}")
    print(f"  Output  : {output_dir}")
    print(f"{'='*55}\n")

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Dataset not found: {raw_dir}")

    all_files        = sorted(os.listdir(raw_dir))
    valid_records    = []
    excluded_records = []
    unparseable      = []

    subject_set    = set()
    emotion_set    = set()
    person_set     = set()
    sequence_set   = set()
    frames_per_emotion   = defaultdict(int)
    frames_per_subject   = defaultdict(int)
    frames_per_sequence  = defaultdict(int)

    print(f"[INFO] Files found: {len(all_files)}")

    for filename in all_files:
        filepath = os.path.join(raw_dir, filename)

        if not os.path.isfile(filepath):
            continue

        # Skip confirmed duplicates
        if is_known_duplicate(filename):
            excluded_records.append({"filename": filename, "reason": "Duplicate (1_AN_mr_ == 01_AN_mr_)"})
            print(f"  [EXCLUDE] {filename}")
            continue

        # Try to decode the filename
        parsed = parse_filename(filename)
        if parsed is None:
            unparseable.append(filename)
            print(f"  [WARN] Unparseable: {filename}")
            continue

        # Verify the file is a readable image
        try:
            from PIL import Image
            with Image.open(filepath) as img:
                width, height = img.size
                color_mode    = img.mode
        except Exception as e:
            excluded_records.append({"filename": filename, "reason": str(e)})
            print(f"  [ERROR] Cannot open: {filename} — {e}")
            continue

        # sequence_id groups all frames of one subject+emotion together
        sequence_id = f"{parsed['subject_id']}_{parsed['emotion_code']}_{parsed['person_code']}"

        valid_records.append({
            "subject_id":   parsed["subject_id"],
            "person_code":  parsed["person_code"],
            "sequence_id":  sequence_id,
            "frame_id":     parsed["frame_id"],
            "emotion_code": parsed["emotion_code"],
            "emotion":      parsed["emotion_name"],
            "image_path":   filepath,
            "width":        width,
            "height":       height,
            "color_mode":   color_mode,
            "filename":     filename,
        })

        subject_set.add(parsed["subject_id"])
        emotion_set.add(parsed["emotion_code"])
        person_set.add(parsed["person_code"])
        sequence_set.add(sequence_id)
        frames_per_emotion[parsed["emotion_name"]]  += 1
        frames_per_subject[parsed["subject_id"]]    += 1
        frames_per_sequence[sequence_id]             += 1

    stats = {
        "total_files_in_dir":  len(all_files),
        "valid_images":        len(valid_records),
        "excluded_duplicates": len(excluded_records),
        "unparseable_files":   unparseable,
        "num_subjects":        len(subject_set),
        "num_emotions":        len(emotion_set),
        "num_persons":         len(person_set),
        "num_sequences":       len(sequence_set),
        "subjects":            sorted(subject_set),
        "emotions":            sorted(emotion_set),
        "persons":             sorted(person_set),
        "sequences":           sorted(sequence_set),
        "frames_per_emotion":  dict(sorted(frames_per_emotion.items())),
        "frames_per_subject":  dict(sorted(frames_per_subject.items())),
        "frames_per_sequence": dict(sorted(frames_per_sequence.items())),
        "excluded_records":    excluded_records,
    }

    return valid_records, stats


def write_metadata_csv(valid_records, csv_path):
    """Write one row per valid image to the metadata CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "subject_id", "person_code", "sequence_id", "frame_id",
        "emotion_code", "emotion", "image_path", "width", "height",
        "color_mode", "filename"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_records)

    print(f"\n[OK] Metadata CSV  -> {csv_path}  ({len(valid_records)} rows)")


def write_statistics_report(stats, report_path):
    """Write a human-readable plain-text statistics report."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    lines = [
        "=" * 60,
        "  KMU-FED Dataset Statistics Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "\n--- OVERVIEW ---",
        f"  Total files in raw dir : {stats['total_files_in_dir']}",
        f"  Valid images           : {stats['valid_images']}",
        f"  Excluded duplicates    : {stats['excluded_duplicates']}",
        f"  Unparseable filenames  : {len(stats['unparseable_files'])}",
        "\n--- DATASET DIMENSIONS ---",
        f"  Subjects   : {stats['num_subjects']}",
        f"  Emotions   : {stats['num_emotions']}",
        f"  Sequences  : {stats['num_sequences']}",
        "\n--- SUBJECTS ---",
    ]

    for s in stats["subjects"]:
        count = stats["frames_per_subject"].get(s, 0)
        lines.append(f"  Subject {s:>4} : {count:>4} frames")

    lines.append("\n--- EMOTION CLASS DISTRIBUTION ---")
    for emotion, count in stats["frames_per_emotion"].items():
        bar = "#" * (count // 5)
        lines.append(f"  {emotion:<12} : {count:>4}  {bar}")

    lines.append("\n--- SEQUENCES ---")
    lines.append(f"  {'Sequence':<25}  {'Frames':>6}")
    lines.append(f"  {'-'*25}  {'-'*6}")
    for seq, count in stats["frames_per_sequence"].items():
        lines.append(f"  {seq:<25}  {count:>6}")

    lines.append("\n--- EXCLUDED FILES ---")
    for rec in stats["excluded_records"]:
        lines.append(f"  {rec['filename']}  ({rec['reason']})")
    if not stats["excluded_records"]:
        lines.append("  None.")

    lines.append("\n--- NOTES ---")
    lines.append("  - Resolution: 1600x1200 RGB (all images)")
    lines.append("  - Not all subjects have all 6 emotions")
    lines.append("  - Sequence-wise split required to prevent data leakage")
    lines.append("  - See explainable.md for full details")
    lines.append("\n" + "=" * 60)

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"[OK] Statistics report -> {report_path}")
    return report_text


def print_summary(stats):
    """Print a concise console summary after the run."""
    print("\n" + "=" * 55)
    print("  Phase 1 Complete")
    print(f"  Valid images : {stats['valid_images']}")
    print(f"  Subjects     : {stats['num_subjects']}")
    print(f"  Emotions     : {stats['num_emotions']}")
    print(f"  Sequences    : {stats['num_sequences']}")
    print(f"  Excluded     : {stats['excluded_duplicates']} duplicates")
    print("\n  Emotion distribution:")
    for emotion, count in stats["frames_per_emotion"].items():
        print(f"    {emotion:<12}: {count}")
    print("=" * 55)


def main():
    args = parse_args()

    valid_records, stats = inspect_dataset(
        raw_dir    = args.raw_dir,
        output_dir = args.output_dir,
    )

    csv_path    = os.path.join(args.output_dir, "kmu_fed_metadata.csv")
    report_path = os.path.join(args.output_dir, "dataset_statistics_report.txt")

    write_metadata_csv(valid_records, csv_path)
    write_statistics_report(stats, report_path)
    print_summary(stats)


if __name__ == "__main__":
    main()
