import torch
import torch.nn as nn

class DilatedNet(nn.Module):
    def __init__(self):
        super(DilatedNet, self).__init__()
        
        # Initial block with standard conv
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # One dilated convolution followed by standard conv
        self.c2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2, bias=False),  # Single dilated conv
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),  # Standard conv
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # One depthwise separable conv followed by standard conv
        self.c3 = nn.Sequential(
            # Single depthwise sep conv
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Standard conv
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # Final conv block
        self.c4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.gap(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x
