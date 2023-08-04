import os
import time
from collections import defaultdict
from typing import Dict, Tuple

from utils import logger


def get_vespa_input_template() -> Dict:
    """
    This method provides the structure and types of the data based on vespa schema.
    """
    default_values = {
        "title": "",
        "body": [],
        "id": "",
        "text_embedding": [],
        "image_embedding": [],
    }

    # defaultdict with a factory function that returns the default value for each key
    return defaultdict(lambda: default_values[None], default_values)


def convert_image_caption_dataset_to_vespa_input_format(
    image_caption_tuple: Tuple[str, str], model
) -> Dict:
    """
    This method converts one (image, caption) to vespa input format based on schema.
    Returns: a dictionary of one converted (caption,image) pair ready to feed vespa
    """

    # get an empty default dictionary based on schema data structure and types
    vespa_input = get_vespa_input_template()

    # add text and image name
    image_path, text_caption = image_caption_tuple
    vespa_input["body"] = [text_caption]
    image_name = os.path.basename(image_path)
    vespa_input["id"] = image_name

    # add text and image embeddings
    vespa_input["text_embedding"] = {0: model.encode_text(text_caption)}
    vespa_input["image_embedding"] = {0: model.encode_image(image_path)}

    return vespa_input


def feed_image_caption_dataset_to_vespa_app(app, image_caption_pairs, model) -> None:
    """
    This method does: 1) convert image-caption to vespa input 2) feed it to vespa
    Args:
        app:
        image_caption_pairs:
        model:

    Returns:
    """
    logger.info("Start sending data samples to the vespa app")

    # convert flickr data to vespa schema format and feed to vespa
    all_elapsed_time = []
    for idx, pair in enumerate(image_caption_pairs):
        # measure how much it takes to create embeddings for one image-caption pair
        start_time = time.time()
        vespa_input = convert_image_caption_dataset_to_vespa_input_format(pair, model)
        end_time = time.time()
        elapsed_time = end_time - start_time
        all_elapsed_time.append(elapsed_time)

        app.feed_data_point(
            schema="document_schema", data_id=str(idx), fields=vespa_input
        )

        if (idx + 1) % 100 == 0:
            print(f"Sent {idx + 1} image-text pairs to vespa ")

    logger.info("Finished sending all the data samples to the vespa app")

    # compute the average time to create embeddings
    ave_time = round(sum(all_elapsed_time) / len(all_elapsed_time), 4)
    print(f"average time to create embeddings for one doc is {ave_time}")


def send_dataset_to_vespa_app(app, dataset, model, app_name):
    """
    This method sends any dataset to vespa based on the vespa app name
    Args:
        app:
        dataset:
        model:
        app_name:
    Returns:
    """

    if app_name == "text2image":
        feed_image_caption_dataset_to_vespa_app(app, dataset, model)

    else:
        raise ValueError(f"Cannot send data to {app_name} vespa app")
