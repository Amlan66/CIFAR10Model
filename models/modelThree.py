import torch
import torch.nn as nn

class DilatedNet(nn.Module):
    def __init__(self):
        super(DilatedNet, self).__init__()
        
        # C1 Block - Initial block with standard conv
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False),  # RF: 3
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # C2 Block - Dilated convolution
        self.c2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, padding=2, dilation=2, bias=False),  # RF: 7
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # C3 Block - Depthwise Separable convolution
        self.c3 = nn.Sequential(
            # Depthwise
            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48, bias=False),  # RF: 15
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # Pointwise
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # C4 Block - Final conv block before GAP
        self.c4 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1, stride=2, bias=False),  # RF: 33
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Global Average Pooling and Final FC layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.gap(x)
        x = x.view(-1, 192)
        x = self.fc(x)
        return x
