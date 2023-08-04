from model.open_clip import BIO_MEDICAL_CLIP, MULTI_LINGUAL_CLIP, OpenClip
from model.sbert_clip import SbertClip
from utils import logger


def get_model(model_name):
    """
    Helper function that returns a model based on the model name.
    """
    logger.info(f"loading the {model_name} model")
    if "openai_clip" in model_name:
        clip_model = SbertClip()
        return clip_model

    elif "laion_clip" in model_name:
        clip_model = OpenClip(**MULTI_LINGUAL_CLIP)
        return clip_model

    elif "biomed_clip" in model_name:
        biomed_clip_model = OpenClip(**BIO_MEDICAL_CLIP)
        return biomed_clip_model

    else:
        raise ValueError(f"Unknown model {model_name}")
