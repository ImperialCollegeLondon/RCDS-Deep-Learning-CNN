import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the linear regression model
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Generate some synthetic data
np.random.seed(42)
X_train = np.random.rand(100, 1) * 10  # Generate 100 random input values
y_train = 2 * X_train + 3 + np.random.randn(100, 1) * 2  # Generate corresponding noisy output values

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Instantiate the model
model = MyNet()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot the fit with increased font sizes for ticks
plt.scatter(X_train, y_train, color='blue', label='Original data')
plt.plot(X_train, model(X_train_tensor).detach().numpy(), color='red', label='Fitted line')
plt.xlabel('X', fontsize=14)
plt.ylabel('y', fontsize=14)
#plt.title('Linear Regression Fit', fontsize=16)
plt.legend(fontsize=12)

# Increase font sizes for ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("linearplot.jpeg")

plt.show()
