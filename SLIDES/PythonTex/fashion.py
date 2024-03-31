import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Load FashionMNIST dataset
dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)

# Define class names for FashionMNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Select random samples from the dataset
num_samples = 16  # 4x4 grid
random_indices = np.random.choice(len(dataset), num_samples, replace=False)

# Create a 4x4 grid plot
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
#fig.suptitle('FashionMNIST Samples', fontsize=16)

for i, ax in enumerate(axes.flat):
    # Get image and label
    image, label = dataset[random_indices[i]]

    # Plot image
    ax.imshow(image, cmap='gray')
    ax.set_title(f'{class_names[label]}', fontsize=16)
    ax.axis('off')

plt.tight_layout()

# Save the plot as an image file
plt.savefig('fashionmnist_samples_4x4.png')

# Show the plot
plt.show()
