import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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

# Predict function
def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in sigmoid(X @ theta)]

# Predictions for the training dataset
predictions = predict(X, theta)

# Calculate the confusion matrix
cm = confusion_matrix(y, predictions)
print(f'Confusion Matrix:\n{cm}')

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions)
recall = recall_score(y, predictions)
f1 = f1_score(y, predictions)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')
