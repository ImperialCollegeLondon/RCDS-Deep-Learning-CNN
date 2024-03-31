import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.linspace(0, 5, 25)
y = 2 * np.sin(X) + np.random.normal(0, 0.5, X.shape)

# Define a simple polynomial regression model
def polynomial_regression(X, y, degree):
    coeffs = np.polyfit(X, y, degree)
    return np.poly1d(coeffs)

# Plot data and model
def plot_model(X, y, model, title, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, label='Data', color='blue')
    x_range = np.linspace(0, 5, 100)
    plt.plot(x_range, model(x_range), label='Model', color='red')
    plt.title(title, fontsize=16)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.ylim(-3.2,3.2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    if save_path:
        plt.savefig(save_path,bbox_inches='tight')
    plt.show()

# Underfitting: Linear model (degree 1)
linear_model = polynomial_regression(X, y, degree=1)
plot_model(X, y, linear_model, 'Underfitting (Linear)', save_path='underfitting.png')

# Overfitting: High-degree polynomial model (degree 15)
high_degree_model = polynomial_regression(X, y, degree=15)
plot_model(X, y, high_degree_model, 'Overfitting (High-degree Polynomial)', save_path='overfitting.png')

# Overfitting: High-degree polynomial model (degree 15)
def true_model(x):
    return 2*np.sin(x)

plot_model(X, y, true_model, 'Sine curve with noise', save_path='fiting.png')

