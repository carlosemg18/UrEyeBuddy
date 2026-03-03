"""
Triplet dataset for face-embedding training.

Expected directory layout (one subfolder per identity):
    data_root/
        person_a/
            img1.jpg
            img2.jpg
        person_b/
            img1.jpg
            ...

Each __getitem__ returns (anchor, positive, negative) tensors.
"""

import os
import random
from pathlib import Path
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


FACE_SIZE = 160  # Input size expected by FaceEmbeddingNet

default_transform = transforms.Compose(
    [
        transforms.Resize((FACE_SIZE, FACE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),  # scales to [0, 1]
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((FACE_SIZE, FACE_SIZE)),
        transforms.ToTensor(),
    ]
)


class TripletFaceDataset(Dataset):
    """Generates online triplets from an identity-folder dataset."""

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform or default_transform

        # Build mapping: identity_label -> [image_paths]
        self.identities: dict[str, list[Path]] = {}
        for person_dir in sorted(self.root.iterdir()):
            if not person_dir.is_dir():
                continue
            imgs = [
                p
                for p in person_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
            if len(imgs) >= 2:  # need at least 2 images for anchor+positive
                self.identities[person_dir.name] = imgs

        self.identity_names = list(self.identities.keys())
        if len(self.identity_names) < 2:
            raise ValueError(
                "Need at least 2 identities with >=2 images each."
            )

        # Flat list for length/indexing — anchor drawn from here
        self._flat: list[Tuple[str, Path]] = []
        for name, paths in self.identities.items():
            for p in paths:
                self._flat.append((name, p))

    def __len__(self) -> int:
        return len(self._flat)

    def __getitem__(self, idx: int):
        identity, anchor_path = self._flat[idx]

        # Positive: another image of the same person
        positives = [p for p in self.identities[identity] if p != anchor_path]
        positive_path = random.choice(positives)

        # Negative: random image of a different person
        neg_identity = random.choice(
            [n for n in self.identity_names if n != identity]
        )
        negative_path = random.choice(self.identities[neg_identity])

        anchor = self.transform(Image.open(anchor_path).convert("RGB"))
        positive = self.transform(Image.open(positive_path).convert("RGB"))
        negative = self.transform(Image.open(negative_path).convert("RGB"))

        return anchor, positive, negative
