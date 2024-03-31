import numpy as np
import matplotlib.pyplot as plt

# Define image dimensions
width = 128
height = 128

# Generate random noise
noise = np.random.rand(height, width)

# Display the noise
plt.imshow(noise, cmap='gray')
plt.axis('off')

# Save the image tightly
plt.savefig('random_noise.png', bbox_inches='tight', pad_inches=0)
