import numpy as np
import matplotlib.pyplot as plt
import random
from net import FeedForwardNetwork, sigmoid, bce_with_logits_loss

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

# normalize the data
X = X / (L/2)

# Convert data to lists for pure Python processing
X_list = X.tolist()
y_list = y.tolist()

# Function to create batches
def create_batches(X, y, batch_size):
    combined = list(zip(X, y))
    random.shuffle(combined)
    batches = []
    for i in range(0, len(combined), batch_size):
        batch = combined[i:i + batch_size]
        X_batch, y_batch = zip(*batch)
        batches.append((X_batch, y_batch))
    return batches

# Parameters
batch_size = 32
num_epochs = 1024
learning_rate = 0.03/batch_size

# Initialize network
model = FeedForwardNetwork(input_size=2, hidden_size=16, learning_rate=learning_rate)

# Enable interactive plotting
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

# Create a grid over the plot area for visualization
x_min, x_max = -1, 1
y_min, y_max = -1, 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    # Create batches
    batches = create_batches(X_list, y_list, batch_size)

    for X_batch, y_batch in batches:
        batch_loss = 0.0
        model.zero_grad()

        for x, y_true in zip(X_batch, y_batch):
            # Forward pass
            output = model.forward(x)
            # Compute loss
            loss = bce_with_logits_loss(output, y_true)
            batch_loss += loss
            # Backward pass
            model.backward(y_true)
            # Update weights

        model.update_weights()
        total_loss += batch_loss

    # print loss every epoch
    avg_loss = total_loss / len(X_list)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    # Visualization phase
    # Compute the model output for each point in the grid
    Z = []
    for point in grid:
        output = model.forward(point.tolist())
        Z.append(sigmoid(output))
    Z = np.array(Z).reshape(xx.shape)

    # Clear the previous plot
    ax.clear()

    # Plot the probabilities as a continuous color map
    cmap = plt.cm.RdBu  # Red to Blue colormap
    im = ax.contourf(xx, yy, 1 - Z, levels=255, cmap=cmap, alpha=0.5, vmin=0, vmax=1)

    # Plot the data points
    ax.scatter(X[y == 0.0, 0], X[y == 0.0, 1], c='blue', edgecolor='k', label='Inside Circle', alpha=0.7)
    ax.scatter(X[y == 1.0, 0], X[y == 1.0, 1], c='red', edgecolor='k', label='Outside Circle', alpha=0.7)
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

