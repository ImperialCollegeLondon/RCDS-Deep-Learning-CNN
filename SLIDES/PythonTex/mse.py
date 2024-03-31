import numpy as np
import matplotlib.pyplot as plt
import os

# Generate true linear function
true_slope = 2
true_intercept = 3
x_true = np.linspace(0, 10, 100)
y_true = true_slope * x_true + true_intercept

# Generate data points around the true linear function
np.random.seed(42)
x_data = np.linspace(0, 10, 10)
y_data = true_slope * x_data + true_intercept + np.random.randn(10)

# Define function to calculate mean squared error
def calculate_mse(slope, intercept):
    y_pred = slope * x_data + intercept
    return np.mean((y_data - y_pred) ** 2)

# Initialize plots
fig, ax = plt.subplots(figsize=(8, 6))

# Iterate through 25 frames
for frame in range(25):
    # Calculate slope and intercept for current frame
    slope = 0.56 + frame * 0.06  # Adjusted for a better initial fit
    intercept = 10.68 - frame * 0.32

    # Plot vertical lines between data points and fitted line
    for x, y in zip(x_data, y_data):
        ax.plot([x, x], [y, slope * x + intercept], color='green', linestyle='--')

    # Plot data points and fitted line
    ax.scatter(x_data, y_data, color='red', label='Data Points')
    ax.plot(x_true, slope * x_true + intercept, color='blue', label='Fitted Line')

    # Calculate and display MSE
    mse = calculate_mse(slope, intercept)
    ax.text(0.75, 0.1, f'MSE = {mse:.2f}', color="green",transform=ax.transAxes, fontsize=16)

    # Set labels and title
    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    
    plt.ylim(0,25)

    # Set legend
    ax.legend(fontsize=12,loc="best")

    # Set tick font size
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Save the plot as an image
    if frame == 24:
        for i in range(10):  # Save the last image with different filenames
            plt.savefig(f'linear_regression_optimization_{frame}_{i}.png')
    else:
        plt.savefig(f'linear_regression_optimization_{frame}.png')

    # Clear the plot for the next iteration
    ax.clear()

# Close the plot
plt.close()

# Create GIF from images
import imageio

# Get the filenames of the saved images
image_files = [f'linear_regression_optimization_{frame}.png' for frame in range(24)] + \
              [f'linear_regression_optimization_24_{i}.png' for i in range(10)]

# Create GIF
with imageio.get_writer('linear_regression_optimization.gif', mode='I') as writer:
    for filename in image_files:
        image = imageio.imread(filename)
        writer.append_data(image)

# Delete the saved images
for filename in image_files:
    os.remove(filename)
