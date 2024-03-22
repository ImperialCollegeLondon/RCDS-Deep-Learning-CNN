import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os

# Load MNIST dataset
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Create directory to save images
os.makedirs("mnist_images", exist_ok=True)

# Function to plot and save ordered pictures
def plot_and_save_ordered_images(data, labels, num_images=10):
    fig, axes = plt.subplots(num_images, num_images, figsize=(10, 10))
    for i in range(num_images):
        for j in range(num_images):
            index = i * num_images + j
            img = data[labels == i][index].numpy().squeeze()
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(f"mnist_images/ordered_images.png")  # Save the plot
    plt.show()

# Plot and save ordered pictures
plot_and_save_ordered_images(train_dataset.data, train_dataset.targets)

# Function to plot and save a random minibatch of images
def plot_and_save_random_minibatch(data, labels, batch_size=10, num_batches=3):
    for batch_num in range(num_batches):
        indices = np.random.randint(0, len(data), size=batch_size)
        fig, axes = plt.subplots(1, batch_size, figsize=(15, 3))
        for i, idx in enumerate(indices):
            img = data[idx].numpy().squeeze()
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(f"mnist_images/minibatch_{batch_num}.png")  # Save the plot
        plt.show()

# Plot and save multiple random minibatches of images
plot_and_save_random_minibatch(train_dataset.data, train_dataset.targets, batch_size=10, num_batches=3)
