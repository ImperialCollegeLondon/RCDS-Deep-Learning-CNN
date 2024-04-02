import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def linear(x):
    return x

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Generate x values
x = np.linspace(-5, 5, 100)

# Compute y values for each activation function
y_linear = linear(x)
y_relu = relu(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)

# Plotting in a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Linear
axs[0, 0].plot(x, y_linear, color='blue')
axs[0, 0].set_title('Linear', fontsize=14, fontweight='bold')
axs[0, 0].tick_params(axis='both', which='major', labelsize=12)
axs[0, 0].set_xlabel('Input (a)', fontsize=12)
axs[0, 0].set_ylabel('Output', fontsize=12)

# ReLU
axs[0, 1].plot(x, y_relu, color='green')
axs[0, 1].set_title('ReLU', fontsize=14, fontweight='bold')
axs[0, 1].tick_params(axis='both', which='major', labelsize=12)
axs[0, 1].set_xlabel('Input (a)', fontsize=12)
axs[0, 1].set_ylabel('Output', fontsize=12)

# Sigmoid
axs[1, 0].plot(x, y_sigmoid, color='red')
axs[1, 0].set_title('Sigmoid', fontsize=14, fontweight='bold')
axs[1, 0].tick_params(axis='both', which='major', labelsize=12)
axs[1, 0].set_xlabel('Input (a)', fontsize=12)
axs[1, 0].set_ylabel('Output', fontsize=12)

# Tanh
axs[1, 1].plot(x, y_tanh, color='purple')
axs[1, 1].set_title('Tanh', fontsize=14, fontweight='bold')
axs[1, 1].tick_params(axis='both', which='major', labelsize=12)
axs[1, 1].set_xlabel('Input (a)', fontsize=12)
axs[1, 1].set_ylabel('Output', fontsize=12)

plt.tight_layout()

# Save the output
plt.savefig('activation_functions_plot.jpeg')

plt.show()
