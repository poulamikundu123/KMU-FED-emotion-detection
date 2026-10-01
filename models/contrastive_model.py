"""
models/contrastive_model.py
Phase 11: Siamese Network architecture for Contrastive Learning.
Features a shared EfficientNet-B0 encoder, 2-layer projection head (1280 -> 512 -> 128),
and Contrastive Loss function with Euclidean margin.
See explainable.md > Phase 11 for details.
"""

import os
import sys
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.efficientnet_encoder import EfficientNetEncoder


class ContrastiveLoss(nn.Module):
    """Contrastive loss function penalizing distant positives and close negatives."""
    def __init__(self, margin=config.CONTRASTIVE_MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, label):
        """Compute contrastive loss between paired embeddings z1 and z2."""
        # Euclidean distance
        euclidean_distance = F.pairwise_distance(z1, z2, keepdim=True)

        # Positive pairs (label=1): pull together; Negative pairs (label=0): push apart up to margin
        loss_pos = label.view(-1, 1) * torch.pow(euclidean_distance, 2)
        loss_neg = (1.0 - label.view(-1, 1)) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss = torch.mean(loss_pos + loss_neg)
        return loss


class SiameseContrastiveNetwork(nn.Module):
    """Siamese Network sharing an EfficientNet encoder with an MLP projection head."""
    def __init__(self, encoder=None, proj_dim=config.PROJECTION_HEAD_DIM, freeze_backbone=config.FREEZE_BACKBONE):
        super().__init__()
        if encoder is None:
            self.encoder = EfficientNetEncoder(pretrained=True)
        else:
            self.encoder = encoder

        if freeze_backbone:
            self.encoder.freeze_backbone()

        # Projection head: 1280 -> 512 -> 128
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

    def forward_one(self, x):
        """Forward pass for a single image, returning L2-normalized 128-dim projection."""
        features = self.encoder(x)
        proj = self.projection_head(features)
        return F.normalize(proj, p=2, dim=1)

    def forward(self, x1, x2):
        """Process a pair of images through the shared encoder and projection head."""
        z1 = self.forward_one(x1)
        z2 = self.forward_one(x2)
        return z1, z2

    def get_encoder(self):
        """Extract the encoder alone for downstream classification."""
        return self.encoder


def run_phase11():
    print("=" * 55)
    print("  KMU-FED Phase 11 — Contrastive Learning Model")
    print(f"  Shared backbone  : {config.EFFICIENTNET_VARIANT}")
    print(f"  Projection head  : 1280 -> 512 -> {config.PROJECTION_HEAD_DIM}")
    print(f"  Contrastive margin: {config.CONTRASTIVE_MARGIN}")
    print("=" * 55)

    # 1. Instantiate Model and Loss
    model = SiameseContrastiveNetwork(freeze_backbone=True)
    criterion = ContrastiveLoss(margin=config.CONTRASTIVE_MARGIN)

    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print(f"[OK] Initialized Siamese Network:")
    print(f"     Total parameters     : {total:,}")
    print(f"     Trainable (Proj Head): {trainable:,} (Backbone is frozen)")

    # 2. Test forward pass with paired dummy inputs
    x1 = torch.randn(4, 3, config.INPUT_SIZE, config.INPUT_SIZE)
    x2 = torch.randn(4, 3, config.INPUT_SIZE, config.INPUT_SIZE)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    z1, z2 = model(x1, x2)

    assert z1.shape == (4, config.PROJECTION_HEAD_DIM), f"Unexpected z1 shape: {z1.shape}"
    assert z2.shape == (4, config.PROJECTION_HEAD_DIM), f"Unexpected z2 shape: {z2.shape}"

    # Verify L2 normalization: norm of each vector should equal 1.0
    norms1 = torch.norm(z1, p=2, dim=1)
    norms2 = torch.norm(z2, p=2, dim=1)
    assert torch.allclose(norms1, torch.ones_like(norms1), atol=1e-5), "z1 is not L2 normalized!"
    assert torch.allclose(norms2, torch.ones_like(norms2), atol=1e-5), "z2 is not L2 normalized!"

    print("\n--- [Forward Pass Verification] ---")
    print(f"Input Pair shapes  : {list(x1.shape)} and {list(x2.shape)}")
    print(f"Embedding 1 shape  : {list(z1.shape)} (L2 norm: {norms1[0].item():.4f})")
    print(f"Embedding 2 shape  : {list(z2.shape)} (L2 norm: {norms2[0].item():.4f})")
    print("--- [Forward Pass Passed] ---")

    # 3. Test loss computation & backward gradient flow
    loss = criterion(z1, z2, labels)
    loss.backward()

    # Verify gradients exist in projection head
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.projection_head.parameters())
    assert has_grads, "No gradients flowed into projection head!"

    print("\n--- [Loss & Backward Verification] ---")
    print(f"Contrastive Loss value : {loss.item():.4f}")
    print(f"Gradient flow check    : PASSED (Gradients computed successfully)")
    print("--- [Loss Test Passed] ---\n")

    print("=" * 55)
    print("  Phase 11 Complete — Siamese Network verified")
    print("=" * 55)


if __name__ == "__main__":
    run_phase11()
