import glob
import os

import numpy as np
import pandas as pd

from torch.utils.data import WeightedRandomSampler
from monai.data.utils import partition_dataset_classes


# prepare data function
def image_name(image_file):
    """Extracts the image name from the image file path."""
    return os.path.basename(image_file).split(".")[0]


def prepare_data(
    images_dir, labels_df, format=".jpg", test_size=0.2, seed=42, warmup=True
):
    """Prepares the data for training and validation."""

    image_files = sorted(glob.glob(os.path.join(images_dir, f"*{format}")))

    files_df = pd.read_csv(labels_df)

    image_list = files_df["image_name"].to_list()
    labels_list = files_df["target"].to_list()
    data_dict = {k: v for k, v in zip(image_list, labels_list)}

    labels_list = [data_dict[image_name(i)] for i in image_files]

    train, val = partition_dataset_classes(
        image_files, labels_list, ratios=[(1 - test_size), test_size], shuffle=True
    )

    if warmup:
        train = train[:500]
        val = val[:500]

    train_dicts = [{"image": i, "label": data_dict[image_name(i)]} for i in train]
    val_dicts = [{"image": i, "label": data_dict[image_name(i)]} for i in val]

    return train_dicts, val_dicts


def weighted_sampler(data, replacement=True):
    """Creates a weighted sampler from a list of dicts"""
    labels = [i["label"] for i in data]
    unique_labels, count = np.unique(labels, return_counts=True)
    label_weight = [sum(count) / c for c in count]
    weights = [label_weight[e] for e in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=replacement)
