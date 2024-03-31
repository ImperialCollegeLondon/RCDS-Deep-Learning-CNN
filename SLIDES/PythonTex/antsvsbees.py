import os
import random
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

# Check if the dataset is already downloaded
dataset_path = "./hymenoptera_data"
if not os.path.exists(dataset_path):
    # Download the dataset
    dataset_url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
    os.system(f"wget {dataset_url}")
    os.system("unzip -q hymenoptera_data.zip")

# Define class names
class_names = ['Ant', 'Bee']

# Load the dataset
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])
dataset = torchvision.datasets.ImageFolder(root='./hymenoptera_data/val', transform=data_transform)

# Select random indices for 5x7 grid
num_rows, num_cols = 4, 11
indices = random.sample(range(len(dataset)), num_rows * num_cols)

# Plot the images
fig, axes = plt.subplots(num_rows, num_cols, figsize=(35, 15))

for i, index in enumerate(indices):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]
    
    image, label = dataset[index]

    # Plot the image
    ax.imshow(image.permute(1, 2, 0))

    # Add a rectangle patch behind the image
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=20, alpha=0.7, edgecolor='green' if class_names[label] == 'Ant' else 'orange',
                             facecolor='none', transform=ax.transAxes)
    ax.add_patch(rect)

    # Add title with class name in green or orange
    ax.set_title(class_names[label], fontsize=25, color='green' if class_names[label] == 'Ant' else 'orange')

    # Hide axis
    ax.axis('off')

plt.tight_layout(pad=0.0)
plt.savefig("beesants.jpeg")
plt.show()
