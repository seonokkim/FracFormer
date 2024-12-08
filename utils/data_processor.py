import os
import numpy as np
import pandas as pd
import torch
import shutil
import random
from utils.data_processor import load_dicom  

def split_train_test(
    df=None,
    train_csv_path=None,
    train_images_path=None,
    test_images_path=None,
    test_csv_path=None,
    test_ratio=0.2,
    random_seed=42
):
    """
    Splits data into training and testing datasets.

    Parameters:
    - df (pd.DataFrame, optional): DataFrame containing StudyInstanceUIDs for splitting.
    - train_csv_path (str, optional): Path to the training CSV file (required if `df` is not provided).
    - train_images_path (str, optional): Directory containing training images (optional).
    - test_images_path (str, optional): Directory where test images will be moved (optional).
    - test_csv_path (str, optional): Path to save the test CSV file (required if `df` is provided).
    - test_ratio (float): Ratio of data to use for testing (default: 0.2).
    - random_seed (int): Seed for reproducibility (default: 42).

    Returns:
    - tuple: (train_df, test_df) if `df` is provided.
    """
    if df is None:
        if train_csv_path is None:
            raise ValueError("Either 'df' or 'train_csv_path' must be provided.")
        df = pd.read_csv(train_csv_path)

    # Set random seed and shuffle the data
    random.seed(random_seed)
    shuffled_indices = list(range(len(df)))
    random.shuffle(shuffled_indices)

    # Split indices for train and test
    test_size = int(len(df) * test_ratio)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    # Split DataFrame
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]

    if train_images_path and test_images_path:
        os.makedirs(test_images_path, exist_ok=True)
        for uid in test_df["StudyInstanceUID"]:
            src_folder = os.path.join(train_images_path, uid)
            dst_folder = os.path.join(test_images_path, uid)
            if os.path.exists(src_folder):
                shutil.move(src_folder, dst_folder)

    if train_csv_path:
        train_df.to_csv(train_csv_path, index=False)

    if test_csv_path:
        test_df.to_csv(test_csv_path, index=False)

    return train_df, test_df


class DataSet(torch.utils.data.Dataset):
    """Custom Dataset for loading DICOM images."""
    def __init__(self, df, path, transforms=None):
        """
        Initialize the dataset.

        Args:
            df (pd.DataFrame): DataFrame containing StudyInstanceUID and Slice information.
            path (str): Path to the DICOM files.
            transforms (callable, optional): Transformations to apply to the images.
        """
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms

    def __getitem__(self, i):
        """
        Get a single item from the dataset.

        Args:
            i (int): Index of the item.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        path = os.path.join(self.path, self.df.iloc[i].StudyInstanceUID, f'{self.df.iloc[i].Slice}.dcm')

        try:
            img = load_dicom(path)[0]
            img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            if self.transforms:
                img = self.transforms(torch.as_tensor(img))
        except Exception as ex:
            print(f"Error loading image at {path}: {ex}")
            return None

        if 'C1_fracture' in self.df:
            frac_targets = torch.as_tensor(
                self.df.iloc[i][['C1_fracture', 'C2_fracture', 'C3_fracture', 'C4_fracture', 'C5_fracture', 'C6_fracture', 'C7_fracture']].astype('float32').values
            )
            vert_targets = torch.as_tensor(
                self.df.iloc[i][['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']].astype('float32').values
            )
            frac_targets = frac_targets * vert_targets  # Enable targets visible on the current slice
            return img, frac_targets, vert_targets
        return img

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.df)
