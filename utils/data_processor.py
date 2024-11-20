import os
import torch
import glob
import re
import pandas as pd
import numpy as np
import pydicom as dicom
import cv2
from torch.utils.data import Dataset



class DataSet(torch.utils.data.Dataset):    
    def __init__(self, df, path, transforms=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms
        
    def __getitem__(self, i):
        path = os.path.join(self.path, self.df.iloc[i].StudyInstanceUID, f'{self.df.iloc[i].Slice}.dcm')        
        
        try:
            img = load_dicom(path)[0]         
            img = np.transpose(img, (2, 0, 1))  # Pytorch uses (batch, channel, height, width) order. Converting (height, width, channel) -> (channel, height, width)
            if self.transforms is not None:
                img = self.transforms(torch.as_tensor(img))
        except Exception as ex:
            print(ex)
            return None
        
        if 'C1_fracture' in self.df:
            frac_targets = torch.as_tensor(self.df.iloc[i][['C1_fracture', 'C2_fracture', 'C3_fracture', 'C4_fracture', 'C5_fracture', 'C6_fracture', 'C7_fracture']].astype('float32').values)
            vert_targets = torch.as_tensor(self.df.iloc[i][['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']].astype('float32').values)
            frac_targets = frac_targets * vert_targets   # we only enable targets that are visible on the current slice
            return img, frac_targets, vert_targets
        return img        
    
    def __len__(self):
        return len(self.df)
    

def load_dicom(path):
    """
    Load and preprocess a DICOM file.
    Converts to RGB format if the image is grayscale.
    """
    # Read the DICOM file
    img = dicom.dcmread(path)
    
    # Access pixel data
    data = img.pixel_array

    # Normalize pixel data to range [0, 255]
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)  # Convert to uint8 for compatibility

    # Convert to RGB format
    if len(data.shape) == 2:  # If the image is grayscale
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    
    return data, img

    

def load_df_test(path):
    df_test = pd.read_csv(f'{path}/test.csv')

    if df_test.iloc[0].row_id == '1.2.826.0.1.3680043.10197_C1':
        # test_images and test.csv are inconsistent in the dev dataset, fixing labels for the dev run.
        df_test = pd.DataFrame({
            "row_id": ['1.2.826.0.1.3680043.22327_C1', '1.2.826.0.1.3680043.25399_C1', '1.2.826.0.1.3680043.5876_C1'],
            "StudyInstanceUID": ['1.2.826.0.1.3680043.22327', '1.2.826.0.1.3680043.25399', '1.2.826.0.1.3680043.5876'],
            "prediction_type": ["C1", "C1", "patient_overall"]}
        )
    return df_test

    
def load_test_slices(path):
    """
    Loads and processes test slices from the given directory.

    Parameters:
        path (str): Path to the directory containing test images.

    Returns:
        pd.DataFrame: A DataFrame with 'StudyInstanceUID' and 'Slice' columns,
                      sorted by 'StudyInstanceUID' and 'Slice'.
    """
    # Find all test slices in the directory
    test_slices = glob.glob(f'{path}/*/*')
    
    # Extract StudyInstanceUID and Slice from file paths
    test_slices = [re.findall(f'{path}/(.*)/(.*).dcm', s)[0] for s in test_slices]
    
    # Create a DataFrame and sort the data
    df_test_slices = (
        pd.DataFrame(data=test_slices, columns=['StudyInstanceUID', 'Slice'])
        .astype({'Slice': int})
        .sort_values(['StudyInstanceUID', 'Slice'])
        .reset_index(drop=True)
    )
    
    return df_test_slices
