# Image/Text → Vector

"""Convenience helpers to produce NumPy vectors from images and text.

These wrappers call the `CLIPModel` methods and convert tensors to
NumPy arrays suitable for FAISS and storage operations.
"""

import numpy as np
from core.preprocess import load_and_preprocess

def embed_image(model, image_path):
    """Load an image, preprocess it for the model, and return a numpy vector.

    Returns an array shaped (1, dim) suitable for adding to FAISS.
    """
    image_tensor = load_and_preprocess(
        image_path,
        model.preprocess,
        model.device
    )
    vector = model.encode_image(image_tensor)
    return vector.cpu().numpy()

def embed_text(model, text):
    """Encode a single text query and return a numpy vector.

    The function wraps the model's `encode_text` and returns a CPU numpy array.
    """
    vector = model.encode_text([text])
    return vector.cpu().numpy()
