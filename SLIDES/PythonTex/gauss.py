import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for the Gaussian distributions
#params = [
#    (1, -1, 1),
#    (0.8, 0, 0.5),
#    (0.8, 2, 0.7),
#    (0.8, -2, 0.8),
#    (0.5, 1.5, 0.4),
#    (0.6, 3, 1),
#    (0.4, -3, 0.6),
#    (0.9, 4, 0.9),
#    (0.8, -4, 0.4)
#]  # (amplitude, mean, std_dev)

params = [(1,0,1)]

# Generate data points for the sum of Gaussian distributions
x = np.linspace(-7, 7, 1000)
y_sum = np.zeros_like(x)

for amplitude, mean, std_dev in params:
    y = amplitude * norm.pdf(x, loc=mean, scale=std_dev)
    y_sum += y

# Plot the sum of Gaussian distributions
plt.plot(x, y_sum, color='blue')

# Fill the area below the curve with color
plt.fill_between(x, y_sum, color='Skyblue', alpha=0.5)

# Add labels and title
plt.xlabel('x')
plt.ylabel('Probability Density')

# Save the plot as an image
plt.savefig('sum_of_gaussians.png')

# Show the plot
plt.show()
