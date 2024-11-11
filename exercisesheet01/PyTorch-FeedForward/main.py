import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from matplotlib.colors import ListedColormap
from net import FeedForwardNetwork
from optimizer import Adam

# Parameters
num_samples = 512   # Total number of samples
radius = 5          # Radius of the circle
center = (0, 0)     # Center of the circle

# Calculate the side length of the square
L = radius * np.sqrt(2 * np.pi)

# Generate random points within the square
X = np.random.uniform(low=-L/2, high=L/2, size=(num_samples, 2))

# Calculate the distance of each point from the center
distances = np.sqrt((X[:, 0] - center[0])**2 + (X[:, 1] - center[1])**2)

# Label points: 0 if inside the circle, 1 if outside (but within the square)
y = np.where(distances <= radius, 0.0, 1.0)

# Convert data to torch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float().unsqueeze(1)  # Labels should be floats and have shape [batch_size, 1]

# Create a TensorDataset and DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
batch_size = 64
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Initialize the network, loss function, and optimizer
model = FeedForwardNetwork(hidden_size=3)
#model = FeedForwardNetwork(hidden_size1=64, hidden_size2=32, hidden_size3=16)
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.01)

# Enable interactive plotting
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

# Create a grid over the plot area for visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.from_numpy(grid).float()

# Training loop
num_epochs = 256
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Visualization phase
    # Get the network's output probabilities over the grid
    model.eval()
    with torch.no_grad():
        outputs = model(grid_tensor)
        probabilities = torch.sigmoid(outputs).numpy().reshape(xx.shape)

    # Clear the previous plot
    ax.clear()
    
    # Plot the probabilities as a continuous color map
    cmap = plt.cm.RdBu  # Red to Blue colormap
    im = ax.contourf(xx, yy, 1 - probabilities, levels=255, cmap=cmap, alpha=0.5, vmin=0, vmax=1)
    
    # Plot the data points
    ax.scatter(X[y == 0.0, 0], X[y == 0.0, 1], c='blue', edgecolor='k', label='Inside Circle', alpha=0.7)
    ax.scatter(X[y == 1.0, 0], X[y == 1.0, 1], c='red', edgecolor='k', label='Outside Circle', alpha=0.7)
    #circle = plt.Circle(center, radius, color='black', fill=False, linewidth=2)
    #ax.add_artist(circle)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'2D Points and Probability Map after Epoch {epoch+1}')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)
    
    # Pause to update the plot
    plt.pause(0.1)

# Disable interactive mode and show the final plot
plt.ioff()
plt.show()

