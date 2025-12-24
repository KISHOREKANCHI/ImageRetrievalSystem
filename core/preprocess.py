# core/preprocess.py
"""Helpers to load images and apply the CLIP preprocessing pipeline.

This module centralizes image loading and device placement so callers
don't need to repeat the same boilerplate.
"""
# Image preprocessing

from PIL import Image
import torch

def load_and_preprocess(image_path, preprocess, device):
    """Load an image from disk and apply model preprocessing.

    Args:
        image_path (str): Path to the image file.
        preprocess (callable): transform returned by `open_clip.create_model_and_transforms`.
        device (str): torch device string (e.g. 'cpu' or 'cuda').

    Returns:
        torch.Tensor: preprocessed image tensor on `device` with batch dim.
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    return image_tensor
