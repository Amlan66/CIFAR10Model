import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                  stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DilatedNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # C1: Regular Conv2d with 5x5 kernel (increased channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),  # Increased from 16 to 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        
        # C2: Dilated Conv2d (increased channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2, dilation=2),  # Increased from 32 to 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        
        # C3: Depthwise Separable Conv (increased channels)
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2, padding=1),  # Increased from 64 to 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        
        # C4: Regular Conv2d (increased channels)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 160, kernel_size=3, stride=2, padding=1),  # Increased from 96 to 160
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.2)
        self.fc = nn.Linear(160, 10)  # Changed from 96 to 160

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.dropout2(x)
        
        x = self.conv4(x)
        x = self.gap(x)
        x = self.dropout3(x)
        x = x.view(-1, 160)  # Changed from 96 to 160
        x = self.fc(x)
        return x