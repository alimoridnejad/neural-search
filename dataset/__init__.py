import os
from typing import List, Tuple

import pandas as pd

from dataset.image_caption_data import (get_coco2014_image_caption_pairs,
                                        get_multi_flickr_image_caption_pairs,
                                        get_roco_image_caption_pairs)
from utils import logger

# data path
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
COCO2014_DIR = os.path.join(DATASET_DIR, "coco2014_test")
ROCO_DIR = os.path.join(DATASET_DIR, "roco_test")
MULTI_FLICKR_DIR = os.path.join(DATASET_DIR, "multi30k_test")


def get_dataset(dataset_name):
    """
    Helper function that returns a dataset object based on dataset name
    """

    logger.info(f"loading {dataset_name} dataset")

    if dataset_name == "flickr_english":
        return get_multi_flickr_image_caption_pairs(MULTI_FLICKR_DIR, dataset_name)

    elif dataset_name == "flickr_french":
        return get_multi_flickr_image_caption_pairs(MULTI_FLICKR_DIR, dataset_name)

    elif dataset_name == "flickr_dutch":
        return get_multi_flickr_image_caption_pairs(MULTI_FLICKR_DIR, dataset_name)

    elif dataset_name == "roco":
        return get_roco_image_caption_pairs(ROCO_DIR)

    elif dataset_name == "coco":
        return get_coco2014_image_caption_pairs(COCO2014_DIR)

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def get_evaluation_labeled_data(image_caption_tuple_list: List[Tuple]) -> pd.DataFrame:
    """
    This method builds IR evaluation data according to Vespa learntorank library
    Args:
        image_caption_tuple_list:
    Returns:
    """

    # create two lists from the tuples
    image_files, captions = map(list, zip(*image_caption_tuple_list))

    logger.info(f"creating labeled dataset of {len(captions)} samples for evaluation")

    # vespa learntorank library needs exactly the following field names and format
    labeled_data = pd.DataFrame(
        {
            "qid": list(range(len(captions))),
            "query": captions,
            "doc_id": [os.path.basename(image_path) for image_path in image_files],
            "relevance": 1,
        }
    )
    return labeled_data
