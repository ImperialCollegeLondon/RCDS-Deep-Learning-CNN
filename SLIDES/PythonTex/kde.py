import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

# Generate synthetic data from two Gaussian distributions
np.random.seed(0)
data1 = np.random.normal(loc=0, scale=1, size=1000)
data2 = np.random.normal(loc=5, scale=2, size=1000)
data = np.concatenate((data1, data2)).reshape(-1, 1)

# Fit a Gaussian Mixture Model to estimate the density
gmm = GaussianMixture(n_components=2)
gmm.fit(data)

# Fit a Kernel Density Estimation model
kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
kde.fit(data)

# Generate points for plotting the estimated densities
x = np.linspace(-5, 15, 1000)
pdf_gmm = np.exp(gmm.score_samples(x.reshape(-1, 1)))
pdf_kde = np.exp(kde.score_samples(x.reshape(-1, 1)))

# Plot the histogram of the data and the estimated densities
plt.hist(data, bins=50, density=True, alpha=0.5, label='Histogram of Data')
plt.plot(x, pdf_gmm, color='red', label='GMM Estimated Density')
plt.plot(x, pdf_kde, color='blue', label='KDE Estimated Density')
plt.xlabel('Value',fontsize=12)
plt.ylabel('Density',fontsize=12)
plt.legend(fontsize=12)
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig("kde.jpeg")
