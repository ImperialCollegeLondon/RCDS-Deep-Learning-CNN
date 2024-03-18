import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

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

# Generate data for plotting
x_vals = np.linspace(-10, 16, 500)
loss_vals = loss_function(x_vals)

# Perform gradient descent optimization
learning_rate = 0.035
num_iterations = 100
start_position = -8
history = gradient_descent(learning_rate, num_iterations, start_position)

# Create plots and save images
image_files = []
for i in range(num_iterations):
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, loss_vals, color='gold', lw =2, label='Loss Function (L)')
    plt.scatter(history[i], loss_function(history[i]), color='firebrick', marker='o', label='Optimization Steps', zorder=5, s=100)
    
    # Plot the gradient line around the moving point
    gradient_x = np.linspace(history[i] - 2, history[i] + 2, 100)
    gradient_y = gradient(history[i]) * (gradient_x - history[i]) + loss_function(history[i])
    plt.plot(gradient_x, gradient_y, color='green', linestyle='--', label='Gradient', zorder=4,lw=2)
    
    plt.plot([history[i], history[i]], [0, loss_function(history[i])], color='firebrick', linestyle=':', zorder=3)
    plt.title('Gradient Descent Optimization', fontweight='bold', fontsize=16)
    plt.xlabel('Parameter Value (w)', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.ylim(0, None)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add text showing position and gradient
    update_equation_text = r'$w_{%d} = w_{%d} - \alpha \cdot \nabla L(w_{%d})$' % (i + 1, i, i)
    position_text = r'$w$ = ${%.2f}$' % history[i]
    gradient_text = r'$\nabla L(w)$ = ${%.2f}$' % gradient(history[i])
    plt.text(0.5, 0.85, position_text, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, color='firebrick')
    plt.text(0.5, 0.80, gradient_text, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, color='green')
    plt.text(0.5, 0.75, update_equation_text, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, color='black')

    # Move legend below the equations
    plt.legend(loc='upper center', fontsize=12, bbox_to_anchor=(0.5, 0.6), fancybox=True, shadow=True, ncol=1)
    
    filename = f'frame_{i:03d}.png'
    plt.savefig(filename)
    plt.close()
    image_files.append(filename)

# Create GIF with reduced frames per second
with imageio.get_writer('gradient_descent.gif', mode='I', fps=5) as writer:
    for filename in image_files:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove image files
for filename in image_files:
    os.remove(filename)
