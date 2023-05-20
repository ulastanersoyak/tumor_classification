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
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),

            #2
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #3
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(),

            #4
            nn.Conv2d(in_channels=40, out_channels=80, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=327680, out_features=50),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=50, out_features=200),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=200, out_features=4),
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