# Image/Text → Vector

import numpy as np
from core.preprocess import load_and_preprocess

def embed_image(model, image_path):
    image_tensor = load_and_preprocess(
        image_path,
        model.preprocess,
        model.device
    )
    vector = model.encode_image(image_tensor)
    return vector.cpu().numpy()

def embed_text(model, text):
    vector = model.encode_text([text])
    return vector.cpu().numpy()
