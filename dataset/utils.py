import re


def clean_caption(caption, max_words):
    """
    This method cleans image caption of public datasets such as coco, flickr, etc.
    """
    # remove non-alphabetic characters
    caption = (
        re.sub(
            r"([,.'!?\"()*#:;~])",
            "",
            caption.lower(),
        )
        .replace("-", " ")
        .replace("/", " ")
        .replace("<person>", "person")
    )

    # remove extra white spaces
    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )

    # removes leading and trailing characters
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    # truncate caption
    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[:max_words])

    if not len(caption):
        raise ValueError("invalid text")

    return caption
