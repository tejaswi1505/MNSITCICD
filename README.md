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

Total Parameters: < 25,000

## Data Augmentation

The training pipeline includes robust data augmentation:
- Random Affine (±15° rotation, ±10% translation)
- Random Perspective (20% distortion)
- Random Erasing (10% probability)
- Normalization (mean=0.1307, std=0.3081)

## Testing Framework

The `test_model.py` includes comprehensive tests:
1. Model Architecture Validation
   - Parameter count check (<25,000)
   - CNN layer structure verification
   - MaxPool placement validation
2. Performance Testing
   - Accuracy threshold (>95%)
3. Documentation Checks
   - README.md existence and completeness

## CI/CD Pipeline

The GitHub Actions workflow (`ml-pipeline.yml`) automates:
1. Environment setup
2. Dependency installation
3. Model training
4. Test execution
5. Validation checks

