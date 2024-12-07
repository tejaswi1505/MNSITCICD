import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import MNISTModel
import datetime

def train():
    # Set device
    device = torch.device("cpu")
    
    # Load MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = MNISTModel().to(device)
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Training Batch: {batch_idx}/{len(train_loader)} Loss: {loss.item():.6f}')
    
    # Save model with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'mnist_model_{timestamp}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")
    
if __name__ == "__main__":
    train() 