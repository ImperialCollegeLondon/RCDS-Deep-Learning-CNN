import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer to generate sine features
class SineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, amplitude=1.0, frequency=1.0, phase=0.0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.amplitude * np.sin(self.frequency * X + self.phase)

# Generate non-linear data with a peak
np.random.seed(0)
X = np.random.uniform(-3, 3, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, size=100)

# Fit a sine curve using Lasso regression
alpha = 0.01
n_realizations = 20
noise_std = 0.7  # Increased noise standard deviation for more variation

lasso = make_pipeline(SineTransformer(), Lasso(alpha=alpha, fit_intercept=False))  # No intercept as sine curve goes through origin
lasso.fit(X, y)

# Bootstrap resampling for multiple realizations with increased noise
realizations = []
for _ in range(n_realizations):
    indices = np.random.choice(len(X), size=len(X), replace=True)
    X_resampled, y_resampled = X[indices], y[indices]
    y_resampled += np.random.normal(0, noise_std, size=len(y_resampled))  # Add noise with increased std
    lasso_resampled = make_pipeline(SineTransformer(), Lasso(alpha=alpha, fit_intercept=False))
    lasso_resampled.fit(X_resampled, y_resampled)
    realizations.append(lasso_resampled)

# Plot the original data and the fitted function
plt.scatter(X, y, color='blue')

X_plot = np.linspace(-3, 3, 1000).reshape(-1, 1)
for i, lasso_model in enumerate(realizations):
    y_plot = lasso_model.predict(X_plot)
    plt.plot(X_plot, y_plot, color='red', alpha=0.5)

plt.xlabel('X')
plt.ylabel('y')

plt.savefig("Low_bias_hig_variance.jpeg")

plt.show()
