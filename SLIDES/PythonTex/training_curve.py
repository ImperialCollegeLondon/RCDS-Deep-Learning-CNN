import numpy as np
import matplotlib.pyplot as plt

# Generate example data
epochs = range(1, 13)
train_loss = [0.9, 0.6, 0.4, 0.3, 0.2, 0.1, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015]
val_loss = [1.0, 0.8, 0.6, 0.5, 0.4, 0.36, 0.35, 0.37, 0.4, 0.44, 0.49, 0.55]

print(len(epochs),len(train_loss),len(val_loss))

# Plotting
fsize = 14
plt.figure(figsize=(10,8))
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.axvline(x=np.argmin(val_loss)+1,linestyle="--",color="grey")
plt.text(5.6, 0.9, 'Underfitting', fontsize=fsize, color='black', ha='center')
plt.arrow(4.6, 0.91, -1, 0, head_width=0.02, head_length=0.1, fc='black', ec='black')
plt.text(8.4, 0.9, 'Overfitting', fontsize=fsize, color='black', ha='center')
plt.arrow(9.3, 0.91, 1, 0, head_width=0.02, head_length=0.1, fc='black', ec='black')
plt.xlabel('Epochs',fontsize=fsize)
plt.ylabel('Loss',fontsize=fsize)
plt.xticks(size=fsize)
plt.yticks(size=fsize)
plt.legend(loc='lower left', frameon=False,fontsize=12)
plt.savefig('training_validation_loss.jpg', bbox_inches='tight')
plt.show()
