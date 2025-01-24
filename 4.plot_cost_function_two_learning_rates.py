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
theta_initial = np.zeros(X.shape[1])
iterations = 100

# Train the model with learning rate 0.1
learning_rate_1 = 0.1
theta_1, cost_history_1 = gradient_descent(X, y, theta_initial.copy(), learning_rate_1, iterations)

# Train the model with learning rate 5
learning_rate_5 = 5
theta_5, cost_history_5 = gradient_descent(X, y, theta_initial.copy(), learning_rate_5, iterations)

# Plot cost function vs. iteration curve for both learning rates
plt.plot(range(iterations), cost_history_1, label='Learning Rate 0.1')
plt.plot(range(iterations), cost_history_5, label='Learning Rate 5')
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.title('Cost Function vs. Iterations for Different Learning Rates')
plt.legend()
plt.grid(True)
plt.show()
