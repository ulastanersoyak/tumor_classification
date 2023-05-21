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

            #1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            #2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            #4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            #1
            nn.Dropout(p=0.5),
            nn.Linear(in_features=131072, out_features=512),
            nn.ReLU(),

            #2
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),

            #3
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=4),
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