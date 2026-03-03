"""
FaceNet-style Siamese network with a MobileNetV2 backbone for producing
compact 128-d face embeddings optimised for on-device inference.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class FaceEmbeddingNet(nn.Module):
    """Produces a 128-dimensional L2-normalised face embedding."""

    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)

        # Use MobileNetV2 features (output: 1280-d after pooling)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 160, 160) face crop tensor, normalised to [0, 1].
        Returns:
            (B, embedding_dim) L2-normalised embedding.
        """
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.embedding(x)
        return nn.functional.normalize(x, p=2, dim=1)


class TripletLoss(nn.Module):
    """Standard triplet-margin loss for metric learning."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss_fn(anchor, positive, negative)
