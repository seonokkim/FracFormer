import torch  
import os  
import cv2
import pydicom as dicom
import numpy as np


def get_device():
    """
    Determines the device to use: 'cuda' if GPU is available, else 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dicom(path):
    """
    Load and preprocess a DICOM file.
    
    Args:
        path (str): Path to the DICOM file.
    
    Returns:
        tuple: A tuple containing:
            - data (numpy.ndarray): Preprocessed image data.
            - img (pydicom.dataset.FileDataset): The original DICOM object.
    """
    img = dicom.dcmread(path)
    data = img.pixel_array
    data = ((data - np.min(data)) / (np.max(data) or 1) * 255).astype(np.uint8)
    if len(data.shape) == 2:  # Grayscale to RGB
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    return data, img


def get_batch_size(device, gpu_batch_size=32, cpu_batch_size=2):
    """
    Determines the batch size based on the device.

    Args:
        device (torch.device): The device being used ('cuda' or 'cpu').
        gpu_batch_size (int): Default batch size for GPU.
        cpu_batch_size (int): Default batch size for CPU.

    Returns:
        int: The appropriate batch size.
    """
    return gpu_batch_size if device.type == 'cuda' else cpu_batch_size

def load_model(model, name, path='.'):
    """
    Loads a PyTorch model's state dictionary from a specified file.

    Args:
        model (torch.nn.Module): The PyTorch model instance to load the state dictionary into.
        name (str): The name of the file (without extension) containing the saved state dictionary.
        path (str): The directory where the file is located. Default is the current directory.

    Returns:
        torch.nn.Module: The model with the loaded state dictionary.
    """
    # Get device
    device = get_device()
    
    # Ensure the model file exists
    file_path = os.path.join(path, f'{name}.pth')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file {file_path} not found.")
    
    # Load the state dictionary
    data = torch.load(file_path, map_location=device)
    model.load_state_dict(data)
    
    return model
