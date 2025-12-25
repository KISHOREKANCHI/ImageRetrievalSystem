import os
from PIL import Image
from tqdm import tqdm
import torch

from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer

from storage.metadata import MetadataStore
from storage.faiss_index import FaissIndex

def run(image_dir, index_path, meta_db):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Models
    caption_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    caption_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_safetensors=True
    ).to(device)


    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    metadata = MetadataStore(meta_db)
    index = FaissIndex(dim=384, path=index_path)

    images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    print(f"📸 Found {len(images)} images")

    for img_path in tqdm(images):
        image = Image.open(img_path).convert("RGB")

        inputs = caption_processor(image, return_tensors="pt").to(device)
        out = caption_model.generate(**inputs, max_new_tokens=30)
        caption = caption_processor.decode(out[0], skip_special_tokens=True)

        vector = embedder.encode([caption], convert_to_numpy=True)

        metadata.add(img_path, caption)
        index.add(vector)

    index.save()
    print("✅ Ingestion complete")

