from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split


class MelanomaDataset(Dataset):
    "SIIM-ISIC Melanoma Dataset"

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
        image = Image.open(file_directory)
        label = self.labels.iloc[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class ISIC2020(Dataset):
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
        image = Image.open(file_directory)
        label = self.labels.iloc[idx]

        if self.transform:
            image = self.transform(image)

        return image, label