import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from matplotlib.animation import FuncAnimation

# XOR inputs and outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])


# Function to plot decision boundary
def plot_decision_boundary(model, X, y, ax, title="Decision Boundary"):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5  # Add some padding
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5  # Add some padding

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    ax.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.coolwarm)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


# Initialize the MLPClassifier
model = MLPClassifier(
    hidden_layer_sizes=(4, 4),  # More neurons and an additional hidden layer
    max_iter=1000,  # More training iterations
    warm_start=True,  # Keep training without resetting the model
    activation='relu',  # ReLU activation
    solver='adam',  # Adam solver
    learning_rate_init=0.001,  # Learning rate
    alpha=0.01  # Regularization term
)

# Set up the plot
fig, ax = plt.subplots(figsize=(6, 5))
epochs = 100  # Total number of epochs to simulate
interval = 0.0005  # Interval between frames (in milliseconds)


# Set up the animation
def update(frame):
    # Fit the model for 'frame' iterations
    model.max_iter = frame
    model.fit(X, y)

    # Clear the axis and plot the new decision boundary
    ax.clear()
    plot_decision_boundary(model, X, y, ax, title=f'Epochs: {frame} / {epochs}')
    ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)


# Create the animation with smaller steps (e.g., training for 1 epoch per frame)
ani = FuncAnimation(fig, update, frames=range(1, epochs + 1, 1), repeat=False, interval=200)

plt.tight_layout()
plt.show()

# Final prediction and accuracy
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Final Accuracy: {accuracy * 100:.2f}%")
