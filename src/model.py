from torch import nn

class tumor_classifier(nn.Module):
    """
    Neural network model for classifying tumor images into 4 classes.
    
    Args:
    None
    
    Attributes:
    layers (nn.Sequential): Sequential container for neural network layers
    
    Methods:
    forward(x): Forward pass through the network
    
    """
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Dropout(p=0.3),
            nn.Linear(in_features=248*248, out_features=64),
            nn.ReLU(),

            nn.Dropout(p=0.3),
            nn.Linear(in_features=64, out_features=4),
            nn.Softmax(dim=1)
        )
        
    def forward(self,x):
        """
        Forward pass through the network
        
        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        return self.layers(x)