import unittest
import torch
from model import MNISTModel
from train import train_model
from unittest.case import TestCase
import os
import platform
from functools import wraps
import time
import torchvision
import torchvision.transforms.v2 as transforms
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def timeout(seconds=10, error_message="Timeout"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if platform.system() == 'Windows':
                # Simple timeout implementation for Windows
                start_time = time.time()
                result = func(*args, **kwargs)
                if time.time() - start_time > seconds:
                    raise TimeoutError(error_message)
                return result
            else:
                # Unix-based systems can use SIGALRM
                import signal
                def _handle_timeout(signum, frame):
                    raise TimeoutError(error_message)

                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result
        return wrapper
    return decorator

class TestMNISTCNN(unittest.TestCase):
    def setUp(self):
        self.model = MNISTModel()
        
    def test_parameter_count(self):
        param_count = sum(p.numel() for p in self.model.parameters())
        self.assertLess(param_count, 25000, 
            f"Model has {param_count} parameters, which exceeds the limit of 25,000")
        print(f"\nParameter count test passed. Total parameters: {param_count}")


    @timeout(300)  # 5 minutes timeout
    def test_model_accuracy(self):
        # Train the model and get accuracy
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MNISTModel().to(device)
        
        # Redirect stdout to capture prints during training
        import sys
        from io import StringIO
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            train_model()
            output = captured_output.getvalue()
        finally:
            sys.stdout = sys.__stdout__

        # Extract accuracy from the output
        accuracy_line = [line for line in output.split('\n') if 'Test Accuracy' in line][0]
        accuracy = float(accuracy_line.split(':')[1].strip('%'))
        
        self.assertGreater(accuracy, 95.0, 
            f"Model accuracy {accuracy:.2f}% is below the required 95%")
        print(f"\nAccuracy test passed. Model achieved {accuracy:.2f}% accuracy")

    def test_model_input_output_shape(self):
        """Test if model handles various batch sizes and maintains correct output shape."""
        batch_sizes = [1, 32, 64, 128]
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, 1, 28, 28)
            output = self.model(test_input)
            
            self.assertEqual(output.shape, (batch_size, 10),
                f"Model output shape {output.shape} is incorrect for batch size {batch_size}. "
                f"Expected ({batch_size}, 10)")
        print("\nInput/Output shape test passed for all batch sizes")

    def test_model_gradients(self):
        """Test if model gradients are properly flowing through all layers."""
        # Forward pass with dummy data
        test_input = torch.randn(1, 1, 28, 28, requires_grad=True)
        output = self.model(test_input)
        loss = output.sum()
        loss.backward()
        
        # Check gradients for each layer
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad,
                f"Gradient is None for layer: {name}")
            self.assertFalse(torch.all(param.grad == 0),
                f"Gradient is all zeros for layer: {name}")
            self.assertFalse(torch.any(torch.isnan(param.grad)),
                f"Gradient contains NaN values for layer: {name}")
        
        print("\nGradient flow test passed for all layers")
    
    def test_readme_exists(self):
        """Test that README.md exists and is not empty."""
        self.assertTrue(os.path.exists('README.md'), "README.md file does not exist")
        
        with open('README.md', 'r') as f:
            content = f.read()
        
        self.assertGreater(len(content), 300, 
            "README.md seems too short. Should contain comprehensive documentation")
        print("\nREADME.md test passed. File exists and contains content")


if __name__ == '__main__':
    unittest.main()