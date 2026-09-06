# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement

Images captured from real-world environments may contain unwanted noise that reduces their quality and affects subsequent image-processing tasks. The objective of this experiment is to develop a Convolutional Autoencoder using deep learning to remove noise from images and reconstruct a cleaner version of the original image.

The model is trained by taking a noisy image as input and using the corresponding original image as the target output. The encoder extracts important features from the noisy image, while the decoder reconstructs the denoised image.

Dataset

The MNIST handwritten digit dataset is used for this experiment. It contains grayscale images of handwritten digits from 0 to 9, with each image having a size of 28 × 28 pixels. Artificial Gaussian noise is added to the images to create noisy input images.


## DESIGN STEPS
### STEP 1: 

Load the MNIST dataset and convert the images into tensors using suitable image transformations.

### STEP 2: 

Add artificial Gaussian noise to the original MNIST images to create noisy images that are used as input to the autoencoder.


### STEP 3: 

Design the convolutional autoencoder consisting of an encoder and decoder. The encoder extracts important features and compresses the input image, while the decoder reconstructs the image.

### STEP 4: 

Initialize the model, define Mean Squared Error (MSE) as the loss function, and use the Adam optimizer for updating the model parameters.

### STEP 5: 

Train the autoencoder using noisy images as input and original clean images as target output. Monitor the reconstruction loss for each epoch.


### STEP 6: 

Evaluate the trained model using test images and visualize the Original, Noisy, and Reconstructed images to verify the denoising performance.


## PROGRAM

### Name: Priya B

### Register Number: 212224230208

```python
# Autoencoder for Image Denoising using PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transform: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])
# Load MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
# Add noise to images
def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)
# Define Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [1,28,28] -> [32,14,14]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [32,14,14] -> [64,7,7]
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [64,7,7] -> [32,14,14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # [32,14,14] -> [1,28,28]
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        # Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()               # Mean Squared Error for reconstruction
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Print model summary
print("Name: NIKILA D\nReg no: 212224230187")
summary(model, input_size=(1, 28, 28))
# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, images)   # Compare denoised output vs original clean image
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: Priya B ")
    print("Register Number:212224230208 ")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)


```

### OUTPUT

## RESULT
Thus, a convolutional autoencoder for image denoising was developed and trained successfully.
