import torch.nn as nn
import torch

class cnn_basic(nn.Module):
    """
    CNN + FNN model for CSI feature extraction and processing
    
    Architecture:
    - Conv2d layer 1: Extract local CSI features from temporal window
    - Conv2d layer 2: Fuse channels
    - Fully connected layers: Map to final 51-dim CSI features
    
    Input: (batch, 1, 2, 51) - 2 time steps of 51 CSI subcarriers
    Output: (batch, 51) - Predicted CSI features
    """
    def __init__(self):
        super(cnn_basic, self).__init__()
        
        # First convolutional layer: extract local features
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(2, 3), padding=(0, 1)),
            nn.LayerNorm([4, 1, 51]),
            nn.ReLU()
        )
        
        # Second convolutional layer: channel fusion
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=(1, 1)),
            nn.LayerNorm([1, 1, 51]),
            nn.ReLU()
        )
        
        # Fully connected network: feature mapping
        self.fnn = nn.Sequential(
            nn.Linear(51, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 51),
            nn.LayerNorm(51),   
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch, 1, 2, 51)
        
        Returns:
            out: Output tensor of shape (batch, 51)
        
        Process:
        1. conv1: Extract local CSI features
        2. conv2: Fuse channels
        3. squeeze: Flatten to vector
        4. fnn: Map to final 51-dim features
        """
        out1 = self.conv1(x)        # (batch, 4, 1, 51)
        out2 = self.conv2(out1)     # (batch, 1, 1, 51)
        out2 = torch.squeeze(out2)  # (batch, 51)
        out = self.fnn(out2)        # (batch, 51)
        return out


if __name__ == "__main__":
    """
    Testing the CNN model
    
    This main block only runs when this file is executed directly,
    not when imported as a module.
    
    Steps:
    1. Create random input: x = torch.rand(1, 1, 2, 51)
       - Simulates CSI feature data
       - Tensor format: (batch=1, channel=1, height=2, width=51)
       - Matches Conv2d input requirements (N, C, H, W)
    
    2. Initialize model: test = cnn_basic()
       - Instantiate the defined CNN+FNN model
    
    3. Forward pass: hold = test(x)
       - Feed random CSI input into model
       - Automatically calls forward(), executing conv1 -> conv2 -> squeeze -> fnn
    
    4. Print output shape: print(hold.size())
       - Display final output shape
       - Expected output size: torch.Size([1, 51]), representing 51-dim feature vector
    """
    
    # Create test input with correct dimensions
    x = torch.rand(1, 1, 2, 51)  # (batch, channel, height, width)
    
    # Initialize model
    test = cnn_basic()
    test.eval()  # Set to evaluation mode
    
    # Forward pass
    with torch.no_grad():
        hold = test(x)
    
    # Display results
    print("="*50)
    print("CNN Model Test")
    print("="*50)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {hold.shape}")
    print(f"Expected:     torch.Size([1, 51])")
    print(f"Match: {'?' if hold.shape == torch.Size([1, 51]) else '?'}")
    print("="*50)