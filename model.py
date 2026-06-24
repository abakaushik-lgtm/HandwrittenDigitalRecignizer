import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        # Convolution Block 1: Conv2D (32 filters, 3x3), ReLU, MaxPool (2x2)
        # Input: 1 x 28 x 28
        # Output after conv1 (with padding=1): 32 x 28 x 28
        # Output after pool1: 32 x 14 x 14
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolution Block 2: Conv2D (64 filters, 3x3), ReLU, MaxPool (2x2)
        # Output after conv2 (with padding=1): 64 x 14 x 14
        # Output after pool2: 64 x 7 x 7
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        # Flattened shape: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Conv Block 1
        x = self.pool1(F.relu(self.conv1(x)))
        # Conv Block 2
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # FC 1 with ReLU and Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # FC 2 (Output logits)
        x = self.fc2(x)
        return x
