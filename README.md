# MNIST Classification with CI/CD Pipeline

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification with a complete CI/CD pipeline using GitHub Actions. The model achieves >95% accuracy while maintaining a lightweight architecture (<25,000 parameters).

## Project Structure 
├── model.py # CNN model architecture definition
├── train.py # Training script with data augmentation
├── test_model.py # Unit tests and model validation
├── .gitignore # Git ignore rules
└── .github/workflows/ml-pipeline.yml # CI/CD pipeline configuration


## Model Architecture

The CNN architecture consists of:

1. **Feature Extraction Layers**:
   - Conv2D Layer 1: 1 → 6 channels (3x3 kernel, padding=1)
   - ReLU Activation
   - MaxPool2D (2x2)
   - Conv2D Layer 2: 6 → 12 channels (3x3 kernel, padding=1)
   - ReLU Activation
   - MaxPool2D (2x2)

2. **Classification Layers**:
   - Flatten Layer
   - Fully Connected: 12 * 7 * 7 → 24 units
   - ReLU Activation
   - Fully Connected: 24 → 10 units (output layer)


## Data Augmentation

The training pipeline includes robust data augmentation:
- Random Affine (±15° rotation, ±10% translation)
- Random Perspective (20% distortion)
- Random Erasing (10% probability)
- Normalization (mean=0.1307, std=0.3081)

## Testing Framework

The `test_model.py` includes comprehensive tests:
1. Parameter Count Validation
   - Check if total number of parameters is less than 25,000
2. Performance Testing
   - Accuracy threshold (>95%)
3. Input/Output Shape Validation
    - Test if model handles various batch sizes and maintains correct output shape
4. Gradient Flow Validation
    - Test if model gradients are properly flowing through all layers
5. Documentation Checks
   - README.md existence and completeness

## CI/CD Pipeline

The GitHub Actions workflow (`ml-pipeline.yml`) automates:
1. Environment setup
2. Dependency installation
3. Model training
4. Test execution
5. Validation checks

