import os
import sys
import torch
import numpy as np
import PIL.Image
import pandas as pd
from sklearn.model_selection import train_test_split

from monai.data import CacheDataset, PersistentDataset
from monai.transforms import Randomizable



class ISIC2020(torch.utils.data.Dataset):
    "ISIC-2020 Dataset"

    def __init__(self,
        csv_file,            # .csv file with image names and labels
        root_dir,            # Root directory of the dataset
        format    = ".jpg",  # Format of the images
        transform = None,    # Transform to be applied to the images
        test      = False,   # Whether to use the test set
        test_size = 0.2,     # Size of the test set
        seed      = 42,      # Random seed
    ):

        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.format = format
        self.transform = transform
        self.test = test
        self.test_size = test_size
        self.seed = seed
        
        # Split the dataset into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.annotations["image_name"],
            self.annotations["target"],
            test_size=self.test_size,
            random_state=self.seed,
            stratify=self.annotations["target"],
        )

        if test:
            self.annotations = self.X_test
            self.labels = self.y_test
        else:
            self.annotations = self.X_train
            self.labels = self.y_train

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        filename = self.annotations.iloc[idx]
        file_directory = self.root_dir + filename + self.format
        image = PIL.Image.open(file_directory)
        label = self.labels.iloc[idx].astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


class CacheISIC2020(CacheDataset):
    "ISIC-2020 Dataset"

    def __init__(self,
        csv_file,                   # .csv file with image names and labels
        root_dir,                   # Root directory of the dataset
        transform,                  # Transform to be applied to the images
        cache_dir   = "cache_dir/", # Directory to cache the dataset
        format      = ".jpg",       # Format of the images
        test        = False,        # Whether to use the test set
        test_size   = 0.2,          # Size of the test set
        cache_rate  = 1.0,
        cache_num   = sys.maxsize,  # Number of images to cache
        seed        = 42,           # Random seed
        num_workers = 8,
        warmup      = False,
        ):

        self.annotations = pd.read_csv(csv_file)
        if warmup:
            self.annotations = self.annotations.iloc[:500]
        self.annotations["image"] = self.annotations["image_name"].apply(lambda x: os.path.join(root_dir, x + format))
        self.root_dir = root_dir
        self.format = format
        self.transform = transform
        self.test = test
        self.test_size = test_size
        self.seed = seed
        
        # Split the dataset into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.annotations["image"],
            self.annotations["target"],
            test_size=self.test_size,
            random_state=self.seed,
            stratify=self.annotations["target"],
        )

        if test:
            data = pd.concat([self.X_test, self.y_test], axis=1).to_dict(orient="records")
        else:
            data = pd.concat([self.X_train, self.y_train], axis=1).to_dict(orient="records")
        super().__init__(
            data, transform=transform, 
            cache_rate=cache_rate, cache_num=cache_num,
            num_workers=num_workers
            )


# class PersistISIC2020(Randomizable, PersistentDataset):
#     "ISIC-2020 Dataset"

#     def __init__(self,
#         csv_file,                  # .csv file with image names and labels
#         root_dir,                  # Root directory of the dataset
#         transform,                 # Transform to be applied to the images
#         cache_dir  = "cache_dir/", # Directory to cache the dataset
#         format     = ".jpg",       # Format of the images
#         test       = False,        # Whether to use the test set
#         test_size  = 0.2,          # Size of the test set
#         cache_rate = 1.0,
#         seed       = 42,           # Random seed
#         ):

#         self.annotations = pd.read_csv(csv_file)
#         self.annotations["image"] = self.annotations["image_name"].apply(lambda x: os.path.join(root_dir, x + format))
#         self.root_dir = root_dir
#         self.format = format
#         self.transform = transform
#         self.test = test
#         self.test_size = test_size
#         self.seed = seed
        
#         # Split the dataset into train and test
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             self.annotations["image"],
#             self.annotations["target"],
#             test_size=self.test_size,
#             random_state=self.seed,
#             stratify=self.annotations["target"],
#         )

#         if test:
            
#             data = pd.concat([self.X_train, self.y_train], axis=1).to_dict(orient="records")
#         else:
#             data = pd.concat([self.X_train, self.y_train], axis=1).to_dict(orient="records")
#         super().__init__(
#             data, transform=transform, 
#             cache_dir=cache_dir
#             )


class PersistISIC2020(PersistentDataset):
    "ISIC-2020 Dataset"

    def __init__(self,
        csv_file,                   # .csv file with image names and labels
        root_dir,                   # Root directory of the dataset
        transform,                  # Transform to be applied to the images
        cache_dir   = "cache_dir/", # Directory to cache the dataset
        format      = ".jpg",       # Format of the images
        test        = False,        # Whether to use the test set
        test_size   = 0.2,          # Size of the test set
        cache_rate  = 1.0,
        cache_num   = sys.maxsize,  # Number of images to cache
        seed        = 42,           # Random seed
        num_workers = 8,
        warmup      = False,
        ):

        self.annotations = pd.read_csv(csv_file)
        if warmup:
            self.annotations = self.annotations.iloc[:500]
        self.annotations["image"] = self.annotations["image_name"].apply(lambda x: os.path.join(root_dir, x + format))
        self.root_dir = root_dir
        self.format = format
        self.transform = transform
        self.test = test
        self.test_size = test_size
        self.seed = seed
        
        # Split the dataset into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.annotations["image"],
            self.annotations["target"],
            test_size=self.test_size,
            random_state=self.seed,
            stratify=self.annotations["target"],
        )

        if test:
            data = pd.concat([self.X_test, self.y_test], axis=1).to_dict(orient="records")
        else:
            data = pd.concat([self.X_train, self.y_train], axis=1).to_dict(orient="records")
        super().__init__(data, transform=transform, cache_dir=cache_dir)