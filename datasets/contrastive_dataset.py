"""
datasets/contrastive_dataset.py
Phase 9: PyTorch Dataset and DataLoader for Contrastive Learning.
Loads pairs on-the-fly and applies dynamic data augmentation during training.
See explainable.md > Phase 9 for details.
"""

import os
import sys
import pandas as pd
from PIL import Image
import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from preprocessing.augmentation import get_training_augmentation
from preprocessing.preprocessing import get_default_transform


class ContrastivePairDataset(Dataset):
    """PyTorch Dataset yielding image pairs and binary similarity labels."""
    def __init__(self, pairs_csv_or_df, is_train=True, transform=None, cache_images=True):
        if isinstance(pairs_csv_or_df, str):
            self.df = pd.read_csv(pairs_csv_or_df)
        else:
            self.df = pairs_csv_or_df.reset_index(drop=True)

        self.is_train = is_train
        if transform is not None:
            self.transform = transform
        elif is_train:
            self.transform = get_training_augmentation(normalize=True)
        else:
            self.transform = get_default_transform()

        # Cache unique images in RAM to eliminate repeated disk I/O
        self.image_cache = {}
        if cache_images:
            unique_paths = set(self.df["image_1"]).union(set(self.df["image_2"]))
            for p in unique_paths:
                if os.path.exists(p):
                    self.image_cache[p] = Image.open(p).convert("RGB")

    def __len__(self):
        return len(self.df)

    def _get_image(self, path):
        if path in self.image_cache:
            return self.image_cache[path]
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img1 = self._get_image(row["image_1"])
        img2 = self._get_image(row["image_2"])
        label = float(row["label"])

        # Apply independent transforms (dynamic augmentation in training)
        tensor1 = self.transform(img1)
        tensor2 = self.transform(img2)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return tensor1, tensor2, label_tensor


def get_contrastive_dataloader(split="train", batch_size=config.CONTRASTIVE_BATCH_SIZE, shuffle=None, num_workers=0):
    """Factory creating configured DataLoader for train, val, or test pairs."""
    csv_map = {
        "train": os.path.join(config.METADATA_DIR, "train_pairs.csv"),
        "val": os.path.join(config.METADATA_DIR, "val_pairs.csv"),
        "test": os.path.join(config.METADATA_DIR, "test_pairs.csv"),
    }

    if split not in csv_map:
        raise ValueError(f"Unknown split '{split}'. Expected one of {list(csv_map.keys())}")

    csv_path = csv_map[split]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing pair CSV for split '{split}': {csv_path}. Run Phase 8 first.")

    is_train = (split == "train")
    if shuffle is None:
        shuffle = is_train

    dataset = ContrastivePairDataset(csv_path, is_train=is_train)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train
    )
    return loader


def run_phase9():
    print("=" * 55)
    print("  KMU-FED Phase 9 — Contrastive Learning Dataset")
    print(f"  Batch size   : {config.CONTRASTIVE_BATCH_SIZE}")
    print(f"  Metadata dir : {config.METADATA_DIR}")
    print("=" * 55)

    # 1. Instantiate Datasets
    train_loader = get_contrastive_dataloader("train", batch_size=config.CONTRASTIVE_BATCH_SIZE)
    val_loader   = get_contrastive_dataloader("val", batch_size=config.CONTRASTIVE_BATCH_SIZE)
    test_loader  = get_contrastive_dataloader("test", batch_size=config.CONTRASTIVE_BATCH_SIZE)

    print(f"[OK] Train DataLoader : {len(train_loader.dataset)} pairs ({len(train_loader)} batches)")
    print(f"[OK] Val DataLoader   : {len(val_loader.dataset)} pairs ({len(val_loader)} batches)")
    print(f"[OK] Test DataLoader  : {len(test_loader.dataset)} pairs ({len(test_loader)} batches)")

    # 2. Test single pair item access
    train_ds = train_loader.dataset
    t1, t2, lbl = train_ds[0]

    assert t1.shape == (3, 224, 224), f"Unexpected shape for view 1: {t1.shape}"
    assert t2.shape == (3, 224, 224), f"Unexpected shape for view 2: {t2.shape}"
    assert lbl.item() in (0.0, 1.0), f"Unexpected label value: {lbl.item()}"

    print("\n--- [Single Pair Verification] ---")
    print(f"View 1 tensor shape : {t1.shape} (dtype: {t1.dtype})")
    print(f"View 2 tensor shape : {t2.shape} (dtype: {t2.dtype})")
    print(f"Pair label tensor   : {lbl.item()} (dtype: {lbl.dtype})")
    print("--- [Single Pair Passed] ---")

    # 3. Test mini-batch yield
    batch_t1, batch_t2, batch_lbl = next(iter(train_loader))
    assert batch_t1.shape == (config.CONTRASTIVE_BATCH_SIZE, 3, 224, 224)
    assert batch_t2.shape == (config.CONTRASTIVE_BATCH_SIZE, 3, 224, 224)
    assert batch_lbl.shape == (config.CONTRASTIVE_BATCH_SIZE,)

    print("\n--- [Mini-Batch Verification] ---")
    print(f"Batch View 1 shape  : {batch_t1.shape}")
    print(f"Batch View 2 shape  : {batch_t2.shape}")
    print(f"Batch Labels shape  : {batch_lbl.shape}")
    print(f"Batch Labels sample : {batch_lbl[:6].tolist()}")
    print("--- [Mini-Batch Passed] ---")

    print("\n" + "=" * 55)
    print("  Phase 9 Complete — Contrastive DataLoaders verified")
    print("=" * 55)


if __name__ == "__main__":
    run_phase9()
