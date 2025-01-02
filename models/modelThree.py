import torch
import torch.nn as nn

class DilatedNet(nn.Module):
    def __init__(self):
        super(DilatedNet, self).__init__()
        
        # C1 Block - Initial block with standard conv
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),  # 32 channels
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # C2 Block - Dilated convolution
        self.c2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2, bias=False),  # 64 channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # C3 Block - Depthwise Separable convolution
        self.c3 = nn.Sequential(
            # Depthwise
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Pointwise
            nn.Conv2d(64, 128, kernel_size=1, bias=False),  # 128 channels
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # C4 Block - Final conv block before GAP
        self.c4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=False),  # Changed to 128 channels
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)  # Changed to 128 input features

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.gap(x)
        x = x.view(-1, 128)  # Changed to 128
        x = self.fc(x)
        return x
