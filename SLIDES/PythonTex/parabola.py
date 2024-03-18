import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio

# Define the loss function (parabola)
def loss_function(x):
    return (x - 3) ** 2 + 5

# Define gradient of the loss function
def gradient(x):
    return 2 * (x - 3)

# Gradient descent optimization
def gradient_descent(learning_rate, num_iterations, start_position):
    x = start_position  # Initial guess
    history = [start_position]
    for _ in range(num_iterations):
        x -= learning_rate * gradient(x)
        history.append(x)
    return history

# Animation function
def animate(frame):
    plt.cla()
    plt.plot(x_vals, loss_vals, color='blue', label='Loss Function')
    plt.scatter(history[frame], loss_function(history[frame]), color='red', marker='o', label='Optimization Steps', zorder=5, s=100)
    plt.plot([history[frame], history[frame]], [0, loss_function(history[frame])], color='black', linestyle='--', zorder=3)
    plt.plot(x_vals, gradient(x_vals), color='green', linestyle='-', label='Gradient', zorder=4)
    plt.title('Gradient Descent Optimization')
    plt.xlabel('Parameter Value')
    plt.ylabel('Loss')
    plt.legend()

# Generate data for plotting
x_vals = np.linspace(-10, 16, 500)
loss_vals = loss_function(x_vals)

# Perform gradient descent optimization
learning_rate = 0.1
num_iterations = 20
start_position = -8
history = gradient_descent(learning_rate, num_iterations, start_position)

# Create animation
fig = plt.figure(figsize=(8, 6))
ani = FuncAnimation(fig, animate, frames=num_iterations, interval=500)

# Save images
image_files = []
for i in range(num_iterations):
    plt.cla()
    plt.plot(x_vals, loss_vals, color='blue', label='Loss Function')
    plt.scatter(history[i], loss_function(history[i]), color='red', marker='o', label='Optimization Steps', zorder=5, s=100)
    plt.plot([history[i], history[i]], [0, loss_function(history[i])], color='black', linestyle='--', zorder=3)
    plt.plot(x_vals, gradient(x_vals), color='green', linestyle='-', label='Gradient', zorder=4)
    plt.title('Gradient Descent Optimization')
    plt.xlabel('Parameter Value')
    plt.ylabel('Loss')
    plt.legend()
    filename = f'frame_{i:02d}.png'
    plt.savefig(filename)
    image_files.append(filename)

# Create GIF
with imageio.get_writer('gradient_descent.gif', mode='I') as writer:
    for filename in image_files:
        image = imageio.imread(filename)
        writer.append_data(image)

# Display animation
plt.show()
