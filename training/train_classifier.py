"""
training/train_classifier.py
Phase 15: Classifier training & fine-tuning with three-stage strategy (Stage A, Stage B, Stage C).
Stage A trains classifier head on cached features.
Stage B optionally fine-tunes top encoder layers end-to-end (only accepted if validation improves).
Stage C performs honest final evaluation on the test set.
See explainable.md > Phase 15 for details.
"""

import os
import sys
import argparse
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from PIL import Image  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torch.utils.data import Dataset, DataLoader, TensorDataset  # type: ignore
from torchvision import transforms  # type: ignore


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.efficientnet_encoder import EfficientNetEncoder
from models.emotion_classifier import EmotionClassifier, EndToEndEmotionModel, get_num_classes_from_metadata
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


def get_mild_training_transform(input_size=config.INPUT_SIZE):
    """Mild augmentation pipeline preserving key facial landmarks for fine-tuning."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


class EmotionImageDataset(Dataset):
    """Memory-cached image dataset for Stage B end-to-end fine-tuning and evaluation."""
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.image_cache = {}

        # Pre-cache images into RAM to eliminate disk I/O bottleneck
        for _, row in self.df.iterrows():
            p = row["processed_path"]
            if p not in self.image_cache and os.path.exists(p):
                self.image_cache[p] = Image.open(p).convert("RGB")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = row["processed_path"]
        img = self.image_cache[p]

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = get_default_transform()(img)

        label = EMOTION_TO_IDX[row["emotion"]]
        return tensor, torch.tensor(label, dtype=torch.long)


def compute_class_weights(labels, num_classes=6, device="cpu"):
    """Compute inverse frequency class weights to address emotion imbalance."""
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * np.maximum(counts, 1).astype(np.float32))
    weights = weights / weights.sum() * num_classes  # Normalize
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate_features(classifier, dataloader, criterion, device):
    """Evaluate classifier performance on cached feature vectors."""
    classifier.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for feats, labels in dataloader:
            feats, labels = feats.to(device), labels.to(device)
            outputs = classifier(feats)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * feats.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1) * 100.0
    return avg_loss, accuracy


def evaluate_end_to_end(model, dataloader, criterion, device):
    """Evaluate end-to-end model on raw image batches."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1) * 100.0
    return avg_loss, accuracy


def train_stage_a(train_features, train_labels, val_features, val_labels,
                  num_classes=6, epochs=config.STAGE_A_EPOCHS,
                  lr=config.STAGE_A_LR, batch_size=config.CLASSIFIER_BATCH_SIZE,
                  device="cpu"):
    """Stage A: Rapid classifier training directly on cached 1280-dim feature vectors."""
    print("\n--- [Stage A] Training Classifier Head on Cached Features ---")
    print(f"  Epochs: {epochs} | LR: {lr} | Batch size: {batch_size}")

    tr_dataset = TensorDataset(torch.from_numpy(train_features).float(), torch.from_numpy(train_labels).long())
    va_dataset = TensorDataset(torch.from_numpy(val_features).float(), torch.from_numpy(val_labels).long())

    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_dataset, batch_size=batch_size, shuffle=False)

    classifier = EmotionClassifier(
        feature_dim=1280,
        num_classes=num_classes,
        hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
        dropout_rate=config.CLASSIFIER_DROPOUT
    ).to(device)

    class_weights = compute_class_weights(train_labels, num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_weights = None
    history = []

    for epoch in range(1, epochs + 1):
        classifier.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for feats, labels in tr_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * feats.size(0)
            preds = torch.argmax(outputs, dim=1)
            tr_correct += (preds == labels).sum().item()
            tr_total += labels.size(0)

        epoch_tr_loss = tr_loss / tr_total
        epoch_tr_acc = tr_correct / tr_total * 100.0

        epoch_va_loss, epoch_va_acc = evaluate_features(classifier, va_loader, criterion, device)
        scheduler.step(epoch_va_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = epoch_va_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_va_loss
            best_val_acc = epoch_va_acc
            best_weights = classifier.state_dict().copy()

        history.append({
            "stage": "Stage_A",
            "epoch": epoch,
            "train_loss": round(epoch_tr_loss, 4),
            "train_acc": round(epoch_tr_acc, 2),
            "val_loss": round(epoch_va_loss, 4),
            "val_acc": round(epoch_va_acc, 2),
            "lr": current_lr,
            "is_best": is_best
        })

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs or is_best:
            best_mark = " [*BEST*]" if is_best else ""
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_tr_loss:.4f} (Acc: {epoch_tr_acc:5.2f}%) | Val Loss: {epoch_va_loss:.4f} (Acc: {epoch_va_acc:5.2f}%){best_mark}")

    # Load and save best Stage A checkpoint
    classifier.load_state_dict(best_weights)
    stage_a_ckpt = os.path.join(config.CHECKPOINTS_DIR, "best_classifier_stageA.pth")
    torch.save(best_weights, stage_a_ckpt)
    print(f"[OK] Best Stage A Val Loss: {best_val_loss:.4f} (Acc: {best_val_acc:.2f}%) saved to {stage_a_ckpt}")

    return classifier, best_val_loss, best_val_acc, history


def train_stage_b(classifier, num_classes=6, epochs=5,
                  lr=config.STAGE_B_LR, batch_size=config.CLASSIFIER_BATCH_SIZE,
                  unfreeze_blocks=1, baseline_val_loss=float("inf"), device="cpu"):
    """Stage B: Conservative end-to-end fine-tuning of top encoder block."""
    print("\n--- [Stage B] End-to-End Fine-Tuning (Top Encoder Block) ---")
    print(f"  Epochs: {epochs} | LR (encoder): {lr} | Unfrozen blocks: {unfreeze_blocks}")

    tr_dataset = EmotionImageDataset(config.TRAIN_CSV, transform=get_mild_training_transform())
    va_dataset = EmotionImageDataset(config.VAL_CSV, transform=get_default_transform())

    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_dataset, batch_size=batch_size, shuffle=False)

    encoder = EfficientNetEncoder(pretrained=False)
    encoder_ckpt = os.path.join(config.CHECKPOINTS_DIR, "contrastive_encoder.pth")
    if os.path.exists(encoder_ckpt):
        encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device, weights_only=False))

    model = EndToEndEmotionModel(encoder=encoder, classifier=classifier, num_classes=num_classes)
    model.to(device)

    # Unfreeze only the top block to preserve features gently
    model.unfreeze_last_n_blocks(n=unfreeze_blocks)
    trainable_enc, total_enc = model.encoder.count_parameters()
    print(f"[OK] Unfrozen encoder parameters: {trainable_enc:,} / {total_enc:,}")

    optimizer = optim.Adam([
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": lr},
        {"params": model.classifier.parameters(), "lr": lr * 5}
    ], weight_decay=1e-4)

    train_labels = np.array([EMOTION_TO_IDX[emo] for emo in tr_dataset.df["emotion"]])
    class_weights = compute_class_weights(train_labels, num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = baseline_val_loss
    best_model_state = None
    improved_over_stage_a = False
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            tr_correct += (preds == labels).sum().item()
            tr_total += labels.size(0)

        epoch_tr_loss = tr_loss / tr_total
        epoch_tr_acc = tr_correct / tr_total * 100.0

        epoch_va_loss, epoch_va_acc = evaluate_end_to_end(model, va_loader, criterion, device)

        is_best = epoch_va_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_va_loss
            best_model_state = model.state_dict().copy()
            improved_over_stage_a = True

        history.append({
            "stage": "Stage_B",
            "epoch": epoch,
            "train_loss": round(epoch_tr_loss, 4),
            "train_acc": round(epoch_tr_acc, 2),
            "val_loss": round(epoch_va_loss, 4),
            "val_acc": round(epoch_va_acc, 2),
            "lr": lr,
            "is_best": is_best
        })

        best_mark = " [*BEST*]" if is_best else ""
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_tr_loss:.4f} (Acc: {epoch_tr_acc:5.2f}%) | Val Loss: {epoch_va_loss:.4f} (Acc: {epoch_va_acc:5.2f}%){best_mark}")

    if improved_over_stage_a and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[OK] Stage B achieved improved validation loss ({best_val_loss:.4f})!")
        return model, history, True
    else:
        print(f"[Info] Stage A model retained as best validation model (Loss: {baseline_val_loss:.4f}).")
        return None, history, False


def evaluate_stage_c(model, num_classes=6, device="cpu"):
    """Stage C: Final honest evaluation on held-out test set."""
    print("\n--- [Stage C] Final Evaluation on Test Set ---")
    te_dataset = EmotionImageDataset(config.TEST_CSV, transform=get_default_transform())
    te_loader = DataLoader(te_dataset, batch_size=config.CLASSIFIER_BATCH_SIZE, shuffle=False)

    test_labels = np.array([EMOTION_TO_IDX[emo] for emo in te_dataset.df["emotion"]])
    class_weights = compute_class_weights(test_labels, num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    test_loss, test_acc = evaluate_end_to_end(model, te_loader, criterion, device)
    print(f"  [Test Results] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}% ({len(te_dataset)} images)")
    return test_loss, test_acc


def run_phase15(epochs_a=config.STAGE_A_EPOCHS, epochs_b=3, lr_a=config.STAGE_A_LR, lr_b=config.STAGE_B_LR, skip_stage_b=False):
    print("=" * 60)
    print("  KMU-FED Phase 15 — Classifier Training & Fine-Tuning")
    print(f"  Stage A Epochs: {epochs_a} | Stage B Epochs: {0 if skip_stage_b else epochs_b}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    # 1. Load cached features from Phase 13
    tr_feats = np.load(os.path.join(config.FEATURES_DIR, "train_features.npy"))
    tr_lbls  = np.load(os.path.join(config.FEATURES_DIR, "train_labels.npy"))
    va_feats = np.load(os.path.join(config.FEATURES_DIR, "val_features.npy"))
    va_lbls  = np.load(os.path.join(config.FEATURES_DIR, "val_labels.npy"))

    num_classes = get_num_classes_from_metadata()

    # 2. Stage A: Train Classifier on Features
    classifier, stage_a_val_loss, stage_a_val_acc, hist_a = train_stage_a(
        tr_feats, tr_lbls, va_feats, va_lbls,
        num_classes=num_classes, epochs=epochs_a, lr=lr_a, device=device
    )

    # Assemble baseline End-to-End Model with Stage A classifier
    encoder = EfficientNetEncoder(pretrained=False)
    encoder_ckpt = os.path.join(config.CHECKPOINTS_DIR, "contrastive_encoder.pth")
    if os.path.exists(encoder_ckpt):
        encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device, weights_only=False))

    final_model = EndToEndEmotionModel(encoder=encoder, classifier=classifier, num_classes=num_classes)
    final_model.to(device)

    # Save baseline unified model
    final_model_path = os.path.join(config.CHECKPOINTS_DIR, "best_emotion_model.pth")
    torch.save(final_model.state_dict(), final_model_path)
    print(f"[OK] Saved unified model checkpoint to {final_model_path}")

    hist_b = []
    # 3. Stage B: Optional Fine-tuning
    if not skip_stage_b and epochs_b > 0:
        stage_b_model, hist_b, improved = train_stage_b(
            classifier, num_classes=num_classes, epochs=epochs_b, lr=lr_b,
            unfreeze_blocks=1, baseline_val_loss=stage_a_val_loss, device=device
        )
        if improved and stage_b_model is not None:
            final_model = stage_b_model
            torch.save(final_model.state_dict(), final_model_path)
            print(f"[OK] Updated unified model checkpoint with Stage B weights.")

    # 4. Stage C: Evaluate on Test Set
    final_model.eval()
    test_loss, test_acc = evaluate_stage_c(final_model, num_classes=num_classes, device=device)

    # 5. Save Training History Log
    all_history = pd.DataFrame(hist_a + hist_b)
    all_history["final_test_acc"] = test_acc
    all_history["final_test_loss"] = test_loss
    history_log_path = os.path.join(config.LOGS_DIR, "classifier_training_history.csv")
    all_history.to_csv(history_log_path, index=False)
    print(f"[OK] Saved complete training history to {history_log_path}")

    print("\n" + "=" * 60)
    print("  Phase 15 Complete — Training Pipeline Finished!")
    print(f"  Stage A Best Val Accuracy: {stage_a_val_acc:.2f}% (Loss: {stage_a_val_loss:.4f})")
    print(f"  Final Test Accuracy:       {test_acc:.2f}% (Loss: {test_loss:.4f})")
    print(f"  Model Checkpoint:          {final_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KMU-FED Phase 15 Classifier Training")
    parser.add_argument("--epochs_a", type=int, default=config.STAGE_A_EPOCHS, help="Epochs for Stage A")
    parser.add_argument("--epochs_b", type=int, default=0, help="Epochs for Stage B (default: 0)")
    parser.add_argument("--lr_a", type=float, default=config.STAGE_A_LR, help="Learning rate for Stage A")
    parser.add_argument("--lr_b", type=float, default=config.STAGE_B_LR, help="Learning rate for Stage B")
    parser.add_argument("--skip_stage_b", action="store_true", help="Skip Stage B fine-tuning")
    args = parser.parse_args()

    run_phase15(epochs_a=args.epochs_a, epochs_b=args.epochs_b, lr_a=args.lr_a, lr_b=args.lr_b, skip_stage_b=args.skip_stage_b)
