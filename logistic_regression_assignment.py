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

# Get the cost function value after convergence
cost_value = cost_history[-1]

print(f'Cost Function Value after Convergence: {cost_value:.4f}')
print(f'Learning Parameters after Convergence: {theta}')

# Plot cost function vs. iterations
plt.plot(range(iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.title('Cost Function vs. Iterations')
plt.show()

# Plot dataset with decision boundary
plt.scatter(X.iloc[:, 1], X.iloc[:, 2], c=y, cmap='rainbow')
x_values = [np.min(X.iloc[:, 1]), np.max(X.iloc[:, 1])]
y_values = -(theta[0] + np.dot(theta[1], x_values)) / theta[2]
plt.plot(x_values, y_values, label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Dataset with Decision Boundary')
plt.show()

# Train the model with different learning rates and plot cost function vs. iteration curve
learning_rate_5 = 5
theta_5, cost_history_5 = gradient_descent(X, y, theta, learning_rate_5, 100)

# Plot cost function vs. iterations for both learning rates
plt.plot(range(100), cost_history[:100], label='Learning Rate 0.1')
plt.plot(range(100), cost_history_5, label='Learning Rate 5')
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.title('Cost Function vs. Iterations for Different Learning Rates')
plt.legend()
plt.show()

# Compute the confusion matrix and other evaluation metrics
def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in sigmoid(X @ theta)]

predictions = predict(X, theta)
confusion_matrix = pd.crosstab(np.array(y).flatten(), np.array(predictions), rownames=['Actual'], colnames=['Predicted'])

TP = confusion_matrix[1][1]
TN = confusion_matrix[0][0]
FP = confusion_matrix[0][1]
FN = confusion_matrix[1][0]

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f'Confusion Matrix:\n{confusion_matrix}')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1_score:.2f}')
