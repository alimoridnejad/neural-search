from PIL import Image
from sentence_transformers import SentenceTransformer


class SbertClip:
    """
    A class to wrap a multilingual clip model using sentence transformers.
    Methods:
    - image encoder: OpenAI "clip-ViT-B-32" trained on 400M (image, english caption) pairs
    - text encoder: SBERT multilingual DistilBERT that maps text in 50+ languages to
      the original "clip-ViT-B-32" vector space.
    ref:
    - https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1
    """

    def __init__(self):
        self.image_model = SentenceTransformer("clip-ViT-B-32")
        self.text_model = SentenceTransformer("clip-ViT-B-32-multilingual-v1")

    def encode_image(self, image):
        image_features = self.image_model.encode(
            Image.open(image), normalize_embeddings=True
        ).tolist()
        return image_features

    def encode_text(self, text):
        text_features = self.text_model.encode(text, normalize_embeddings=True).tolist()
        return text_features
