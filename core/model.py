# Load CLIP model

"""Wrapper around an OpenCLIP model providing simple encode helpers.

The `CLIPModel` class loads the chosen CLIP variant and exposes
`encode_image` and `encode_text` which return L2-normalized embeddings.
"""

import torch
import open_clip


class CLIPModel:
    """Load CLIP weights and provide convenience encode methods.

    Args:
        model_name (str): model identifier (e.g. "ViT-B-32").
        pretrained (str): pretrained weights spec for `open_clip`.
    """
    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        # choose device automatically
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # create model + preprocess pipeline + tokenizer
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # move model to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode_image(self, image_tensor):
        """Encode an image tensor and return an L2-normalized feature tensor."""
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def encode_text(self, texts):
        """Tokenize and encode text(s), returning L2-normalized features.

        `texts` should be an iterable of strings; the tokenizer handles batching.
        """
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features
