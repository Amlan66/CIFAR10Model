import torch
import torch.nn as nn

class DilatedNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # C1: Initial block with standard conv
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # C2: One dilated convolution followed by standard conv
        self.c2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, padding=2, dilation=2, bias=False),
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
            nn.Conv2d(48, 72, kernel_size=1, bias=False),  # Reduced to 72
            nn.BatchNorm2d(72),
            nn.ReLU(),
            # Standard conv
            nn.Conv2d(72, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # C4: Final conv block
        self.c4 = nn.Sequential(
            nn.Conv2d(72, 72, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Conv2d(72, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(72, 10)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.gap(x)
        x = x.view(-1, 72)
        x = self.fc(x)
        return x
