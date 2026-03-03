#!/usr/bin/env python3
"""
Convert a trained FaceEmbeddingNet checkpoint to a quantised Core ML model.

Usage:
    python export_coreml.py --checkpoint face_embedding.pth \
                            --output FaceEmbedding.mlpackage
"""

import argparse

import torch
import coremltools as ct
from coremltools.models.neural_network import quantization_utils

from model import FaceEmbeddingNet
from dataset import FACE_SIZE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export model to Core ML")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="face_embedding.pth",
        help="Path to trained .pth weights",
    )
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument(
        "--output", type=str, default="FaceEmbedding.mlpackage"
    )
    p.add_argument(
        "--quantize",
        action="store_true",
        default=True,
        help="Apply float16 quantisation (default: True)",
    )
    return p.parse_args()


def export(args: argparse.Namespace) -> None:
    # Load trained model
    model = FaceEmbeddingNet(
        embedding_dim=args.embedding_dim, pretrained=False
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    # Trace with dummy input
    dummy = torch.randn(1, 3, FACE_SIZE, FACE_SIZE)
    traced = torch.jit.trace(model, dummy)

    # Convert to Core ML
    ml_model = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="faceImage",
                shape=(1, 3, FACE_SIZE, FACE_SIZE),
                scale=1.0 / 255.0,
                bias=[0.0, 0.0, 0.0],
            )
        ],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.iOS16,
    )

    # Apply float16 quantisation to reduce size and speed up NPU inference
    if args.quantize:
        ml_model = quantization_utils.quantize_weights(
            ml_model, nbits=16
        )

    ml_model.author = "UrEyeBuddy"
    ml_model.short_description = (
        "128-d face embedding model for on-device recognition."
    )
    ml_model.save(args.output)
    print(f"Core ML model saved → {args.output}")


if __name__ == "__main__":
    export(parse_args())
