"""
training/train_contrastive.py
Phase 12: Train Siamese Contrastive Network using Contrastive Loss.
Monitors validation loss, adjusts learning rate, and saves best model checkpoints.
See explainable.md > Phase 12 for details.
"""

import os
import sys
import time
import argparse
import pandas as pd  # type: ignore
import torch  # type: ignore
import torch.optim as optim  # type: ignore


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.contrastive_model import SiameseContrastiveNetwork, ContrastiveLoss
from datasets.contrastive_dataset import get_contrastive_dataloader


def train_one_epoch(model, dataloader, criterion, optimizer, device, max_batches=None):
    """Run one training epoch over pair batches."""
    model.train()
    # Keep encoder in eval mode to preserve frozen BatchNorm statistics
    model.encoder.eval()

    running_loss = 0.0
    total_batches = 0

    for idx, (x1, x2, labels) in enumerate(dataloader):
        if max_batches and idx >= max_batches:
            break

        x1 = x1.to(device)
        x2 = x2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        z1, z2 = model(x1, x2)
        loss = criterion(z1, z2, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

    return running_loss / max(total_batches, 1)


def evaluate(model, dataloader, criterion, device, max_batches=None):
    """Compute mean contrastive loss over validation pairs."""
    model.eval()
    running_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for idx, (x1, x2, labels) in enumerate(dataloader):
            if max_batches and idx >= max_batches:
                break

            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)

            z1, z2 = model(x1, x2)
            loss = criterion(z1, z2, labels)

            running_loss += loss.item()
            total_batches += 1

    return running_loss / max(total_batches, 1)


def run_phase12(epochs=5, batch_size=config.CONTRASTIVE_BATCH_SIZE, lr=5e-4, max_batches=None):
    print("=" * 55)
    print("  KMU-FED Phase 12 — Contrastive Training")
    print(f"  Target epochs    : {epochs}")
    print(f"  Batch size       : {batch_size}")
    print(f"  Learning rate    : {lr}")
    print(f"  Checkpoints dir  : {config.CHECKPOINTS_DIR}")
    print("=" * 55)

    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # Initialize model, criterion, and optimizer
    model = SiameseContrastiveNetwork(freeze_backbone=True).to(device)
    criterion = ContrastiveLoss(margin=config.CONTRASTIVE_MARGIN).to(device)
    optimizer = optim.AdamW(model.projection_head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # DataLoaders
    train_loader = get_contrastive_dataloader("train", batch_size=batch_size, shuffle=True)
    val_loader   = get_contrastive_dataloader("val", batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    history = []

    best_model_path = os.path.join(config.CHECKPOINTS_DIR, "best_contrastive_model.pth")
    encoder_path = os.path.join(config.CHECKPOINTS_DIR, "contrastive_encoder.pth")
    history_csv = os.path.join(config.LOGS_DIR, "contrastive_training_history.csv")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, max_batches=max_batches)
        val_loss = evaluate(model, val_loader, criterion, device, max_batches=max_batches)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_dur = time.time() - t_epoch_start

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            # Save Siamese model checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, best_model_path)
            # Save standalone encoder for downstream feature extraction
            torch.save(model.get_encoder().state_dict(), encoder_path)

        star = " *" if is_best else ""
        print(f"  Epoch [{epoch:2d}/{epochs:2d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_dur:.1f}s{star}", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "lr": current_lr,
            "duration_sec": round(epoch_dur, 1),
            "is_best": is_best
        })

    total_time = time.time() - start_time
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv, index=False)

    print("\n" + "=" * 55)
    print("  Phase 12 Complete")
    print(f"  Total time       : {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best Val Loss    : {best_val_loss:.4f}")
    print(f"  Best Model Check : {best_model_path}")
    print(f"  Encoder Check    : {encoder_path}")
    print(f"  History Log      : {history_csv}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 12: Contrastive Training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.CONTRASTIVE_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches per epoch for quick test")
    args = parser.parse_args()

    run_phase12(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_batches=args.max_batches)
