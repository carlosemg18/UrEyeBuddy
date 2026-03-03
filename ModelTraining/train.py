#!/usr/bin/env python3
"""
Train the FaceEmbeddingNet with triplet loss.

Usage:
    python train.py --data_dir ./data/faces --epochs 30 --batch_size 32
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import FaceEmbeddingNet, TripletLoss
from dataset import TripletFaceDataset, default_transform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train face-embedding model")
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root folder with identity sub-folders",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--output", type=str, default="face_embedding.pth")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    dataset = TripletFaceDataset(args.data_dir, transform=default_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    model = FaceEmbeddingNet(
        embedding_dim=args.embedding_dim, pretrained=True
    ).to(device)
    criterion = TripletLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        bar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for anchor, positive, negative in bar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        epoch_loss = running_loss / len(loader)
        print(f"  Epoch {epoch} — avg loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), args.output)
            print(f"  Saved best model → {args.output}")

    print(f"Training complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    train(parse_args())
