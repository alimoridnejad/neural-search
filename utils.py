import logging
import random
from argparse import ArgumentParser, ArgumentTypeError

import numpy as np
import torch
import yaml


def get_logger(log_file=None):
    """
    This method setup the logger.
    Returns:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create a formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S%p"
    )

    # create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # create a file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = get_logger()


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def get_args_parser():
    """
    This method returns common parameters for different performance test.
    Returns: ArgumentParser
    """
    parser = ArgumentParser(description="common experiment parameters")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", default=20, type=int, help="set a random seed")
    parser.add_argument(
        "--config_file", type=str, default=None, help="Path to the configuration file."
    )
    parser.add_argument(
        "--save_results",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to save metric results for later analysis.",
    )
    parser.add_argument("--output_dir", type=str, default="./output/")
    return parser


def print_config(config):
    """
    This method prints the config parameters in the terminal.
    """
    print("\n=== Configuration Parameters ===")
    for param_name, param_value in config.items():
        print(f"{param_name}: {param_value}")
    print("================================\n")


def read_config(config_file):
    """
    This method loads a yaml config file and returns it.
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def seed_all(seed):
    """
    This method sets a seed to ensure reproducibility of results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
