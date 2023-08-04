import open_clip
import torch
from PIL import Image

# Pre-trained model names
MULTI_LINGUAL_CLIP = {
    "model_name": "xlm-roberta-base-ViT-B-32",
    "pretrained": "laion5b_s13b_b90k",
}
BIO_MEDICAL_CLIP = {
    "model_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "pretrained": None,
}


class OpenClip:
    """
    A class to wrap a clip model using Open Clip Community
    Based on the application, choose a model name listed in below links:
    - https://huggingface.co/models?library=open_clip
    - https://github.com/mlfoundations/open_clip
    """

    def __init__(self, model_name, pretrained=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_image(self, image):
        with torch.no_grad():
            image = self.preprocess(Image.open(image)).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image, normalize=True)
            image_features = image_features.squeeze(0).tolist()
        return image_features

    def encode_text(self, text):
        with torch.no_grad():
            text = self.tokenizer([text], context_length=256).to(self.device)
            text_features = self.model.encode_text(text, normalize=True)
            text_features = text_features.squeeze(0).tolist()
        return text_features
