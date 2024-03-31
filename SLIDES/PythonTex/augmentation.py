import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import numpy as np

# Download the dataset
dataset_url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"

# Define transformations for image augmentation
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize the image to 224x224
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Randomly translate the image
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # Adjust brightness, contrast, saturation, and hue
    transforms.RandomRotation(30),  # Randomly rotate the image by up to 30 degrees
    transforms.RandomResizedCrop(512, scale=(0.7, 1.0), ratio=(0.7, 1.4)),  # Randomly crop and resize the image
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

# Load the dataset
dataset = ImageFolder(root="hymenoptera_data", transform=None)  # Load images without applying transformation

# Select a random original image from the dataset
random_index = random.randint(0, len(dataset) - 1)
original_image, _ = dataset[random_index]

# Apply transformations to the original image
transformed_image = transform(original_image)

# Create a list to store augmented images
augmented_images = []

# Apply the same transformations to the original image multiple times
for _ in range(8):
    augmented_images.append(transform(original_image).permute(1, 2, 0))  # Rearrange dimensions for Matplotlib

# Create a subplot grid
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

# Plot the original image
axes[0, 0].imshow(original_image)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

# Plot the augmented images
for i, ax in enumerate(axes.flat[1:]):
    ax.imshow(augmented_images[i])
    ax.set_title(f'Augmented {i+1}')
    ax.axis('off')

# Adjust layout
plt.tight_layout()

# Save the plot as an image file
plt.savefig('augmented_images_grid.png')

# Show the plot
plt.show()
