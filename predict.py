"""
predict.py
Interactive single-image emotion inference utility.
Accepts any raw photo or cropped face, runs end-to-end emotion recognition,
and outputs predicted emotion, confidence score, and full class probability breakdown.
Enhanced with Phase 4 Affine Eye Alignment and NIR Domain Adaptation.
"""

import os
import sys

# Suppress unraisable lz4 buffer flush errors on Anaconda exit
def _suppress_unraisable(unraisable):
    if unraisable.exc_type is ValueError and "I/O operation on closed file" in str(unraisable.exc_value):
        return
    sys.__unraisablehook__(unraisable)

if hasattr(sys, "unraisablehook"):
    sys.unraisablehook = _suppress_unraisable

# Suppress TensorFlow logging noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch  # type: ignore


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models.efficientnet_encoder import EfficientNetEncoder
from models.emotion_classifier import EmotionClassifier, EndToEndEmotionModel
from preprocessing.preprocessing import get_default_transform
from preprocessing.face_alignment import align_face, compute_eye_alignment_angle


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


def load_model(device="cpu"):
    """Load trained EndToEndEmotionModel from checkpoints."""
    encoder = EfficientNetEncoder(pretrained=False)
    encoder_path = os.path.join(config.CHECKPOINTS_DIR, "contrastive_encoder.pth")
    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=False))

    classifier = EmotionClassifier(feature_dim=1280, num_classes=len(CLASS_NAMES))
    cls_path = os.path.join(config.CHECKPOINTS_DIR, "best_classifier_stageA.pth")
    if os.path.exists(cls_path):
        classifier.load_state_dict(torch.load(cls_path, map_location=device, weights_only=False))

    model = EndToEndEmotionModel(encoder=encoder, classifier=classifier, num_classes=len(CLASS_NAMES))

    # Fallback to unified model if available
    unified_path = os.path.join(config.CHECKPOINTS_DIR, "best_emotion_model.pth")
    if os.path.exists(unified_path):
        try:
            model.load_state_dict(torch.load(unified_path, map_location=device, weights_only=False))
        except Exception:
            pass

    model.to(device)
    model.eval()
    return model


def detect_align_and_crop_face(pil_img, align=True, match_domain=False, margin_ratio=0.10):
    """
    Detect primary face region with MTCNN, optionally apply Phase 4 affine
    eye alignment (leveling eye axis to horizontal), and optionally apply
    NIR domain adaptation for webcam photos.
    """
    try:
        from mtcnn import MTCNN  # type: ignore
        detector = MTCNN()
        img_np = np.array(pil_img.convert("RGB"))
        detections = detector.detect_faces(img_np)
        if detections:
            # Pick largest detected face
            best = max(detections, key=lambda d: d["box"][2] * d["box"][3])
            bx, by, bw, bh = best["box"]
            kp = best.get("keypoints", {})
            angle = 0.0

            # Phase 4 Eye Alignment
            if align and "left_eye" in kp and "right_eye" in kp:
                angle = compute_eye_alignment_angle(
                    kp["left_eye"][0], kp["left_eye"][1],
                    kp["right_eye"][0], kp["right_eye"][1]
                )
                center = (
                    (kp["left_eye"][0] + kp["right_eye"][0]) / 2.0,
                    (kp["left_eye"][1] + kp["right_eye"][1]) / 2.0
                )
                img_np = align_face(img_np, angle, center=center)
                re_det = detector.detect_faces(img_np)
                if re_det:
                    best = max(re_det, key=lambda d: d["box"][2] * d["box"][3])
                    bx, by, bw, bh = best["box"]

            h, w = img_np.shape[:2]
            pad_w = int(bw * margin_ratio)
            pad_h = int(bh * margin_ratio)
            x1 = max(0, bx - pad_w)
            y1 = max(0, by - pad_h)
            x2 = min(w, bx + bw + pad_w)
            y2 = min(h, by + bh + pad_h)
            cropped_np = img_np[y1:y2, x1:x2]

            # Domain Adaptation: convert to monochrome and normalize contrast (KMU-FED NIR profile)
            if match_domain:
                gray = cv2.cvtColor(cropped_np, cv2.COLOR_RGB2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                norm_gray = clahe.apply(gray)
                cropped_np = cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2RGB)

            return Image.fromarray(cropped_np), True, angle
    except Exception:
        pass
    return pil_img, False, 0.0


def predict_emotion(image_path, model=None, device="cpu", auto_crop=True, align=True, match_domain=False):
    """Run emotion prediction pipeline on single image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if model is None:
        model = load_model(device=device)

    raw_img = Image.open(image_path).convert("RGB")

    # If already a preprocessed face from KMU-FED data directory, skip auto-crop
    norm_path = os.path.normpath(image_path).lower()
    if "processed" in norm_path or "aligned_faces" in norm_path or "detected_faces" in norm_path:
        auto_crop = False

    angle = 0.0
    if auto_crop:
        cropped_img, was_cropped, angle = detect_align_and_crop_face(raw_img, align=align, match_domain=match_domain)
    else:
        cropped_img, was_cropped = raw_img, False

    transform = get_default_transform(input_size=config.INPUT_SIZE)
    img_tensor = transform(cropped_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_emotion = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100.0

    return {
        "image_path": image_path,
        "predicted_emotion": pred_emotion,
        "predicted_idx": pred_idx,
        "confidence": confidence,
        "probabilities": {name: float(probs[i]) * 100.0 for i, name in enumerate(CLASS_NAMES)},
        "was_cropped": was_cropped,
        "alignment_angle": angle,
        "domain_matched": match_domain,
        "cropped_image": cropped_img,
    }


def print_prediction_results(result):
    """Print formatted terminal report with ASCII bar chart."""
    print("\n" + "=" * 62)
    print("           KMU-FED Facial Emotion Recognition Result          ")
    print("=" * 62)
    print(f"  Input Image : {result['image_path']}")
    if result["was_cropped"]:
        align_msg = f" (Eye-Aligned by {result['alignment_angle']:+.1f} deg)" if abs(result["alignment_angle"]) > 0.1 else ""
        domain_msg = " [NIR Domain Match: ON]" if result.get("domain_matched") else ""
        print(f"  Face Detected: Yes (Auto-cropped{align_msg}{domain_msg})")
    print("-" * 62)
    print(f"  >>> PREDICTED EMOTION : {result['predicted_emotion'].upper()} ({result['confidence']:.2f}% Confidence) <<<")
    print("-" * 62)
    print("  Full Class Probability Distribution:")
    for name in CLASS_NAMES:
        prob = result["probabilities"][name]
        bar_len = int(prob / 100.0 * 25)
        bar = "#" * bar_len + "-" * (25 - bar_len)
        star = " *" if name == result["predicted_emotion"] else "  "
        print(f"    {name:<10} : {prob:6.2f}%  [{bar}]{star}")
    print("=" * 62 + "\n")


def annotate_and_save(result, output_path="results/prediction_annotated.jpg"):
    """Draw prediction banner on top of image and save to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = result["cropped_image"].copy()
    draw = ImageDraw.Draw(img)

    banner_text = f"{result['predicted_emotion']} ({result['confidence']:.1f}%)"
    w, h = img.size

    # Draw banner rectangle at the bottom
    banner_h = max(35, int(h * 0.12))
    draw.rectangle([0, h - banner_h, w, h], fill=(0, 0, 0, 200))
    draw.text((10, h - banner_h + 8), banner_text, fill=(255, 255, 255))

    img.save(output_path)
    print(f"[OK] Saved annotated result image to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict facial emotion on any image")
    parser.add_argument("--image", type=str, default=None, help="Path to input face image")
    parser.add_argument("--no_crop", action="store_true", help="Disable automatic MTCNN face cropping")
    parser.add_argument("--no_align", action="store_true", help="Disable Phase 4 affine eye alignment")
    parser.add_argument("--domain_match", "--gray", action="store_true", help="Apply monochrome NIR domain adaptation for webcam photos")
    parser.add_argument("--save_annotated", action="store_true", help="Save image with prediction banner")
    parser.add_argument("--output", type=str, default="results/prediction_annotated.jpg", help="Annotated output path")
    args = parser.parse_args()

    # Default to an exemplary test face if none supplied
    if args.image is None:
        sample_path = None
        if os.path.exists(config.TEST_CSV):
            df_test = pd.read_csv(config.TEST_CSV)
            if "processed_path" in df_test.columns:
                sample_path = df_test.iloc[0]["processed_path"]

        if sample_path is None or not os.path.exists(sample_path):
            for root, _, files in os.walk(config.PROCESSED_DIR):
                for f in files:
                    if f.endswith(".jpg"):
                        sample_path = os.path.join(root, f)
                        break
                if sample_path:
                    break

        print(f"[Info] No --image supplied. Using test sample: {sample_path}")
        args.image = sample_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = predict_emotion(
        args.image,
        device=device,
        auto_crop=not args.no_crop,
        align=not args.no_align,
        match_domain=args.domain_match,
    )
    print_prediction_results(result)

    if args.save_annotated:
        annotate_and_save(result, output_path=args.output)


if __name__ == "__main__":
    main()
