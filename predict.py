"""
predict.py
Interactive single-image emotion inference utility.
Accepts any raw photo or cropped face, runs end-to-end emotion recognition,
and outputs predicted emotion, confidence score, and full class probability breakdown.
"""

import os
# Suppress TensorFlow logging noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch  # type: ignore


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models.efficientnet_encoder import EfficientNetEncoder
from models.emotion_classifier import EmotionClassifier, EndToEndEmotionModel
from preprocessing.preprocessing import get_default_transform


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


def detect_and_crop_face(pil_img):
    """Optionally detect and crop primary face region using MTCNN if present."""
    try:
        from mtcnn import MTCNN  # type: ignore
        detector = MTCNN()
        img_np = np.array(pil_img.convert("RGB"))
        detections = detector.detect_faces(img_np)
        if detections:
            # Pick largest face
            best = max(detections, key=lambda d: d["box"][2] * d["box"][3])
            bx, by, bw, bh = best["box"]
            pad_w = int(bw * 0.10)
            pad_h = int(bh * 0.10)
            w, h = pil_img.size
            x1 = max(0, bx - pad_w)
            y1 = max(0, by - pad_h)
            x2 = min(w, bx + bw + pad_w)
            y2 = min(h, by + bh + pad_h)
            return pil_img.crop((x1, y1, x2, y2)), True
    except Exception:
        pass
    return pil_img, False


def predict_emotion(image_path, model=None, device="cpu", auto_crop=True):
    """Run emotion prediction pipeline on single image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if model is None:
        model = load_model(device=device)

    raw_img = Image.open(image_path).convert("RGB")

    # If already a preprocessed face, skip MTCNN
    norm_path = os.path.normpath(image_path).lower()
    if "processed" in norm_path or "aligned_faces" in norm_path or "detected_faces" in norm_path:
        auto_crop = False

    cropped_img, was_cropped = detect_and_crop_face(raw_img) if auto_crop else (raw_img, False)

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
        "cropped_image": cropped_img,
    }


def print_prediction_results(result):
    """Print formatted terminal report with ASCII bar chart."""
    print("\n" + "=" * 62)
    print("           KMU-FED Facial Emotion Recognition Result          ")
    print("=" * 62)
    print(f"  Input Image : {result['image_path']}")
    if result["was_cropped"]:
        print("  Face Detected: Yes (Auto-cropped to face region)")
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
    result = predict_emotion(args.image, device=device, auto_crop=not args.no_crop)
    print_prediction_results(result)

    if args.save_annotated:
        annotate_and_save(result, output_path=args.output)


if __name__ == "__main__":
    main()
