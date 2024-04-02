import numpy as np
import matplotlib.pyplot as plt

# Generate balanced data
num_samples_balanced = 10000
balanced_data = 1+np.random.randint(0, 2, size=num_samples_balanced)

# Generate unbalanced data
num_samples_unbalanced = 10000
unbalanced_data = 1+np.random.choice([0, 1, 1], size=num_samples_unbalanced, p=[0.9, 0.05, 0.05])

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# Define bin edges for balanced and unbalanced data
bin_edges_balanced = [0.75, 1.25, 1.75, 2.25]  # Centered at 1 and 2
bin_edges_unbalanced = [0.75, 1.25, 1.75, 2.25]  # Centered at 1 and 2

# Balanced data plot
axes[0].hist(balanced_data, bins=bin_edges_balanced, color='orange', alpha=0.7)
axes[0].set_title('Balanced Data',fontsize=12)
axes[0].set_xlabel('Class',fontsize=12)
axes[0].set_ylabel('Frequency',fontsize=12)
axes[0].set_xticks([1, 2])  # Adjusting xticks
axes[0].tick_params(axis='both', labelsize=12)

# Unbalanced data plot
axes[1].hist(unbalanced_data, bins=bin_edges_unbalanced, color='firebrick', alpha=0.7)
axes[1].set_title('Imbalanced Data',fontsize=12)
axes[1].set_xlabel('Class',fontsize=12)
axes[1].set_ylabel('Frequency',fontsize=12)
axes[1].set_xticks([1, 2])  # Adjusting xticks
axes[1].tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.savefig("balance.jpeg")
plt.show()
