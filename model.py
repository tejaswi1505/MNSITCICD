import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

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
def show_images(images, title):
    """Display a batch of images."""
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(np.transpose(make_grid(images, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.axis('off')

def save_augmentation_examples(dataset, transform, n_samples=8):
    """Save examples of original and augmented images."""
    # Select random samples
    indices = torch.randperm(len(dataset))[:n_samples]
    
    # Get original images and their labels
    original_images = torch.stack([dataset[i][0] for i in indices])
    labels = [dataset[i][1] for i in indices]
    
    # Apply augmentations multiple times to the same images
    augmented_images1 = torch.stack([
        transform(dataset[i][0]) for i in indices
    ])
    augmented_images2 = torch.stack([
        transform(dataset[i][0]) for i in indices
    ])
    
    # Create a single figure
    fig = plt.figure(figsize=(15, 12))
    
    # Plot original images
    ax1 = fig.add_subplot(3, 1, 1)
    plt.title(f'Original Images (Labels: {labels})')
    plt.imshow(np.transpose(make_grid(original_images, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.axis('off')
    
    # Plot first set of augmented images
    ax2 = fig.add_subplot(3, 1, 2)
    plt.title('Augmented Version 1')
    plt.imshow(np.transpose(make_grid(augmented_images1, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.axis('off')
    
    # Plot second set of augmented images
    ax3 = fig.add_subplot(3, 1, 3)
    plt.title('Augmented Version 2')
    plt.imshow(np.transpose(make_grid(augmented_images2, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', bbox_inches='tight', pad_inches=0.5)
    plt.close()