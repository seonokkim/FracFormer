import torch  # PyTorch core library
import timm  # PyTorch Image Models library for ViT model

class VertebraeNet(torch.nn.Module):
    """
    VertebraeNet: A model for vertebrae prediction using a pre-trained Vision Transformer (ViT).
    """
    def __init__(self, pretrained=True):
        """
        Initialize the VertebraeNet model.

        Args:
            pretrained (bool): Whether to use pre-trained weights for the ViT model.
        """
        super().__init__()
        # Load a ViT base model from timm
        self.model = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
        
        # Define a fully connected layer for vertebrae predictions
        # Input size is 1000 (ViT output size), and output size is 7 (number of vertebrae classes)
        self.nn_vertebrae = torch.nn.Sequential(
            torch.nn.Linear(1000, 7),  # Fully connected layer
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor (e.g., image data).

        Returns:
            torch.Tensor: Raw logits for vertebrae predictions.
        """
        # Pass input through the ViT model
        x = self.model(x)
        
        # Pass ViT output through the vertebrae-specific fully connected layer
        return self.nn_vertebrae(x)

    def predict(self, x):
        """
        Perform a forward pass and apply sigmoid activation for predictions.

        Args:
            x (torch.Tensor): Input tensor (e.g., image data).

        Returns:
            torch.Tensor: Vertebrae probabilities (values between 0 and 1).
        """
        # Get raw logits from the forward method
        logits = self.forward(x)
        
        # Apply sigmoid activation to convert logits into probabilities
        return torch.sigmoid(logits)
