import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import MNISTModel
import pytest
import glob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_latest_model():
    model_files = glob.glob('mnist_model_*.pth')
    if not model_files:
        raise FileNotFoundError("No trained model files found. Please run train.py first.")
    latest_model = max(model_files)
    logger.info(f"Using model: {latest_model}")
    return latest_model

def test_model_architecture():
    model = MNISTModel()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params}")
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_model_accuracy():
    device = torch.device("cpu")
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Load model
    model = MNISTModel().to(device)
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    logger.info(f"Model accuracy: {accuracy:.2f}%")
    assert accuracy > 95, f"Model accuracy is {accuracy:.2f}%, should be > 95%"

if __name__ == "__main__":
    pytest.main([__file__]) 