# Load CLIP model

import torch
import open_clip

class CLIPModel:
    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

    def encode_image(self, image_tensor):
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def encode_text(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features
