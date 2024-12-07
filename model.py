import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
            super(MNISTModel, self).__init__()
            self.features = nn.Sequential(
                # First conv layer: 1 -> 6 channels, 3x3 kernel
                nn.Conv2d(1, 6, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Second conv layer: 6 -> 12 channels, 3x3 kernel
                nn.Conv2d(6, 12, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            
            # Fully connected layers
            self.classifier = nn.Sequential(
                nn.Linear(12 * 7 * 7, 24),
                nn.ReLU(),
                nn.Linear(24, 10)
            )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
