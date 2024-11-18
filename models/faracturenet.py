import torch  # PyTorch core library
import torchvision as tv  # PyTorch library for models and datasets
from torchvision.models.feature_extraction import create_feature_extractor  # For feature extraction from models

class fracturenet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Load a pre-trained Swin-B model from torchvision
        tv_model = tv.models.swin_b(weights=tv.models.Swin_B_Weights.DEFAULT)
        
        # Create a feature extractor to extract intermediate features from the 'flatten' layer
        self.model = create_feature_extractor(tv_model, ['flatten'])
        
        # Define a fully connected layer for fracture predictions
        # Input size is 1024 (feature size), and output size is 7 (number of classes)
        self.nn_fracture = torch.nn.Sequential(
            torch.nn.Linear(1024, 7),  # Fully connected layer
        )
        
        # Define a fully connected layer for vertebrae predictions
        # Input size is 1024 (feature size), and output size is 7 (number of classes)
        self.nn_vertebrae = torch.nn.Sequential(
            torch.nn.Linear(1024, 7),  # Fully connected layer
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor (e.g., image data).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Fracture logits (raw, unnormalized scores for each class)
                - Vertebrae logits (raw, unnormalized scores for each class)
        """
        # Extract features from the input using the Swin-B model
        x = self.model(x)['flatten']
        
        # Compute fracture and vertebrae predictions
        return self.nn_fracture(x), self.nn_vertebrae(x)

    def predict(self, x):
        """
        Perform a forward pass and apply sigmoid activation for predictions.

        Args:
            x (torch.Tensor): Input tensor (e.g., image data).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Fracture probabilities (values between 0 and 1)
                - Vertebrae probabilities (values between 0 and 1)
        """
        # Get logits from the forward method
        frac, vert = self.forward(x)
        
        # Apply sigmoid activation to convert logits into probabilities
        return torch.sigmoid(frac), torch.sigmoid(vert)
