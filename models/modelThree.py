import torch
import torch.nn as nn

class DilatedNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # C1: Initial block with standard conv (reduced channels)
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False),  # Reduced to 24 channels
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # C2: One dilated convolution followed by standard conv
        self.c2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, padding=2, dilation=2, bias=False),  # Reduced to 48 channels
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # C3: One depthwise separable conv followed by standard conv
        self.c3 = nn.Sequential(
            # Depthwise sep conv
            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 96, kernel_size=1, bias=False),  # Reduced to 96 channels
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # Standard conv
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # C4: Final conv block
        self.c4 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1, stride=2, bias=False),  # Reduced to 96 channels
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, 10)  # Changed to 96

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.gap(x)
        x = x.view(-1, 96)
        x = self.fc(x)
        return x
