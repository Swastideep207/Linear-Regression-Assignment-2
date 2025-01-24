import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
X = pd.read_csv("c:/Users/KIIT/Documents/6th SEM/AD LAB/24.01.25/logisticX.csv", header=None)
y = pd.read_csv("c:/Users/KIIT/Documents/6th SEM/AD LAB/24.01.25/logisticY.csv", header=None)

# Ensure y is a 1D array
y = y.values.flatten()

# Normalize the independent variables
X = (X - X.mean()) / X.std()

# Add intercept term to X
X.insert(0, 'Intercept', 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    return (-1 / m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        theta -= (learning_rate / m) * (X.T @ (sigmoid(X @ theta) - y))
        cost_history.append(cost_function(X, y, theta))
    return theta, cost_history

# Initialize variables
theta = np.zeros(X.shape[1])
learning_rate = 0.1
iterations = 1000

# Train the model
theta, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)

# Ensure theta has the expected number of elements
if len(theta) == 3:
    # Plot the given dataset with different colors for different classes
    for i in range(len(y)):
        if y[i] == 0:
            plt.plot(X.iloc[i, 1], X.iloc[i, 2], 'ro')  # Red dots for class 0
        else:
            plt.plot(X.iloc[i, 1], X.iloc[i, 2], 'bo')  # Blue dots for class 1

    # Plot the decision boundary
    x_values = [np.min(X.iloc[:, 1]), np.max(X.iloc[:, 1])]
    y_values = -(theta[0] + np.dot(theta[1], x_values)) / theta[2]
    plt.plot(x_values, y_values, label='Decision Boundary')
else:
    print("Error: Unexpected size of theta. Expected 3, got", len(theta))

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Dataset with Decision Boundary')
plt.show()
