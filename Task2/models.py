"""
models.py
---------
Model building blocks for Task 2.

  build_encoder       — ImageNet-pretrained encoder (ResNet-18/50 or EfficientNet-B0)
  SupConModel         — encoder + MLP projection head for SupCon pre-training
  EncoderClassifier   — frozen/unfrozen encoder + linear head for classification
  supcon_loss         — Supervised Contrastive Loss (Khosla et al., 2020)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import config

SUPPORTED_BACKBONES = ["resnet18", "resnet50", "efficientnet_b0"]


def build_encoder(backbone: str, pretrained: bool = True):
    """
    Build an encoder from the chosen backbone.
    Returns (encoder, feat_dim) where encoder maps (B,3,H,W) → (B, feat_dim).
    """
    if backbone == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        return nn.Sequential(*list(m.children())[:-1]), 512

    elif backbone == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet50(weights=w)
        return nn.Sequential(*list(m.children())[:-1]), 2048

    elif backbone == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        return nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1)), 1280

    else:
        raise ValueError(f"Unknown backbone '{backbone}'. "
                         f"Choose from {SUPPORTED_BACKBONES}.")


class SupConModel(nn.Module):
    """Encoder + MLP projection head for Supervised Contrastive pre-training."""

    def __init__(self, backbone: str):
        super().__init__()
        self.encoder, feat_dim = build_encoder(backbone, pretrained=True)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config.SC_PROJ_DIM),
        )

    def forward(self, x):
        h = self.encoder(x).flatten(1)
        return F.normalize(self.projector(h), dim=1)


class EncoderClassifier(nn.Module):
    """SupCon encoder (frozen or partially unfrozen) + linear classification head."""

    def __init__(self, encoder: nn.Module, feat_dim: int, num_classes: int):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x).flatten(1))


def supcon_loss(features, labels, temperature=config.SC_TEMPERATURE):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    features : L2-normalised embeddings, shape (2N, d)
    labels   : integer class labels, shape (2N,)

    Positives are all views sharing the same ground-truth label.
    The per-class normalisation in the denominator handles class imbalance.
    """
    N      = features.size(0)
    device = features.device
    sim    = torch.mm(features, features.T) / temperature

    labels   = labels.view(-1, 1)
    pos_mask = torch.eq(labels, labels.T).float().to(device)
    eye      = torch.eye(N, device=device)
    pos_mask = pos_mask - eye          # exclude self-similarity

    neg_mask   = 1 - eye
    sim_max, _ = torch.max(sim * neg_mask, dim=1, keepdim=True)
    sim        = sim - sim_max.detach()  # numerical stability

    exp_sim  = torch.exp(sim) * neg_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    n_pos         = pos_mask.sum(1)
    mean_log_prob = (pos_mask * log_prob).sum(1) / (n_pos + 1e-8)
    # Only average over anchors that have at least one positive
    return -mean_log_prob[n_pos > 0].mean()
