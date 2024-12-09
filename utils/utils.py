import torch  # For loading the model and handling tensors
import os  # For file path manipulations
import cv2  # For image processing
import pydicom as dicom  # For handling DICOM files
import numpy as np  # For numerical operations
import gc  # For garbage collection

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


def save_model(name, model):
    """
    Save a PyTorch model's state dictionary to a specified file.
    
    Args:
        name (str): The name of the file (without extension) to save the model.
        model (torch.nn.Module): The PyTorch model to save.
    """
    torch.save(model.state_dict(), f'{name}.pth')


def filter_nones(batch):
    """
    Filters out None values from a batch of data.

    Args:
        batch (list): A list of data samples, where some samples may be None.

    Returns:
        list: A list of non-None data samples.
    """
    return torch.utils.data.default_collate([item for item in batch if item is not None])


def gc_collect():
    """
    Collect garbage and clear CUDA cache to free memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
