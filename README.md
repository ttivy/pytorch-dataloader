# pytorch-dataloader
Custom DataLoader for PyTorch
# Installation
`pip install git+https://github.com/ttivy/pytorch-dataloader`
# Usage
```Python
from dataloader import StepDataLoader # This package
from torchvision.datasets import MNIST

# Number of samples in each epoch (Not steps)
NUM_SAMPLES = 100

dataset = MNIST('./mnist')
loader = StepDataLoader(dataset, num_samples=NUM_SAMPLES)
print(len(loader)) # NUM_SAMPLES // batch_size
```
