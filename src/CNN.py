import torch
import torch.nn as nn

class cnn_basic(nn.Module):
    def __init__(self, F, dropout_rate=0.25):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1,3), padding=(0,1)),
            nn.LayerNorm([4,1,F]),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=(1,1)),
            nn.LayerNorm([1,1,F]),
            nn.ReLU()
        )

        self.fnn = nn.Sequential(
            nn.Linear(F, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(32, F)  # Regression output no Sigmoid
        )

    def forward(self, x):
        # x: [B, F] or [B,1,1,F]
        if len(x.shape) == 2:
            x = x.unsqueeze(1).unsqueeze(1)  # -> [B,1,1,F]

        h = self.conv1(x)  # -> [B,4,1,F]
        h = self.conv2(h)  # -> [B,1,1,F]
        h = h.flatten(start_dim=1)  # -> [B, F]
        return self.fnn(h)
