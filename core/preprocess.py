# Image preprocessing

from PIL import Image
import torch

def load_and_preprocess(image_path, preprocess, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    return image_tensor
