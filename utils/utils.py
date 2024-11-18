import torch  # For loading the model and handling tensors
import os  # For file path manipulations

def get_device():
    """
    Determines the device to use: 'cuda' if GPU is available, else 'cpu'.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch_size(device, gpu_batch_size=32, cpu_batch_size=2):
    """
    Determines the batch size based on the device.

    Args:
        device (str): The device being used ('cuda' or 'cpu').
        gpu_batch_size (int): Default batch size for GPU.
        cpu_batch_size (int): Default batch size for CPU.

    Returns:
        int: The appropriate batch size.
    """
    return gpu_batch_size if device == 'cuda' else cpu_batch_size

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
    # Dynamically get the device to map the model's state dictionary
    device = get_device()
    
    # Load the state dictionary from the .tph file
    data = torch.load(os.path.join(path, f'{name}.tph'), map_location=device)
    
    # Load the state dictionary into the model
    model.load_state_dict(data)
    
    return model
