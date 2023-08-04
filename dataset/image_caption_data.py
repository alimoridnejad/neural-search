import os


def get_multi_flickr_image_caption_pairs(data_dir, dataset_name):
    """
    This method returns 1k image-caption pairs from multilingual flickr.
    Args:
        data_dir: directory of multi flickr files
        dataset_name:
    Returns:
    """
    if dataset_name == "flickr_english":
        captions_file = os.path.join(data_dir, "test_2017_flickr.en")

    elif dataset_name == "flickr_french":
        captions_file = os.path.join(data_dir, "test_2017_flickr.fr")

    elif dataset_name == "flickr_dutch":
        captions_file = os.path.join(data_dir, "test_2017_flickr.de")

    else:
        raise ValueError(f"Unknown flickr dataset name {dataset_name}")

    # get text captions
    with open(captions_file, "r") as file:
        captions = [caption.rstrip() for caption in file]

    # get image files
    image_ids_file = os.path.join(data_dir, "test_2017_flickr.txt")
    with open(image_ids_file, "r") as file:
        img_dir = os.path.join(data_dir, "test_2017-flickr-images")
        image_files = [os.path.join(img_dir, img_id.rstrip()) for img_id in file]

    # create (image, caption) tuple list
    image_caption_tuple_list = list(zip(image_files[:5], captions[:5]))

    return image_caption_tuple_list


def get_roco_image_caption_pairs(data_dir):
    """
    This method returns 8k image-caption pairs from roco test dataset.
    Args:
        data_dir: directory of roco files
    Returns:
    """

    # get text captions
    captions_file = os.path.join(data_dir, "captions.txt")
    with open(captions_file, "r") as file:
        captions = [caption.rstrip() for caption in file]

    # get image files
    image_ids_file = os.path.join(data_dir, "image_ids.txt")
    with open(image_ids_file, "r") as file:
        img_dir = os.path.join(data_dir, "images")
        image_files = [os.path.join(img_dir, img_id.rstrip()) for img_id in file]

    # create (image, caption) tuple list
    image_caption_tuple_list = list(zip(image_files, captions))

    return image_caption_tuple_list


def get_coco2014_image_caption_pairs(data_dir):
    """
    This method returns 5k image-caption pairs from coco test dataset.
    Args:
        data_dir: directory of coco files
    Returns:
    """

    # get text captions
    captions_file = os.path.join(data_dir, "captions.txt")
    with open(captions_file, "r") as file:
        captions = [caption.rstrip() for caption in file]

    # get image files
    image_ids_file = os.path.join(data_dir, "image_ids.txt")
    with open(image_ids_file, "r") as file:
        img_dir = os.path.join(data_dir, "images")
        image_files = [os.path.join(img_dir, img_id.rstrip()) for img_id in file]

    # create (image, caption) tuple list
    image_caption_tuple_list = list(zip(image_files, captions))

    return image_caption_tuple_list
