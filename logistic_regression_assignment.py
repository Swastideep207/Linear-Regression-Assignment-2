import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

class LogisticRegressionGD:
    def __init__(self, learning_rate=0.1, decay_rate=0.01, iterations=1000, lambda_=0.1):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.iterations = iterations
        self.lambda_ = lambda_
        self.theta = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost_function(self, X, y):
        m = len(y)
        h = self.sigmoid(X @ self.theta)
        reg_term = (self.lambda_ / (2 * m)) * np.sum(np.square(self.theta[1:]))
        return (-1 / m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)) + reg_term
    
    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        cost_history = []
        
        for i in range(self.iterations):
            learning_rate = self.learning_rate / (1 + self.decay_rate * i)  # Learning rate decay
            gradient = (X.T @ (self.sigmoid(X @ self.theta) - y)) / m
            gradient[1:] += (self.lambda_ / m) * self.theta[1:]  # L2 Regularization
            self.theta -= learning_rate * gradient
            cost_history.append(self.cost_function(X, y))
        
        return cost_history
    
    def predict(self, X):
        return [1 if x >= 0.5 else 0 for x in self.sigmoid(X @ self.theta)]

def cross_validate(X, y, model, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        accuracies.append(accuracy)
    
    return np.mean(accuracies)

# Load dataset
X = pd.read_csv("c:/Users/KIIT/Documents/6th SEM/AD LAB/24.01.25/logisticX.csv", header=None)
y = pd.read_csv("c:/Users/KIIT/Documents/6th SEM/AD LAB/24.01.25/logisticY.csv", header=None).values.flatten()

# Normalize X
X = (X - X.mean()) / X.std()
X.insert(0, 'Intercept', 1)

# Train the model
model = LogisticRegressionGD(learning_rate=0.1, iterations=1000)
cost_history = model.fit(X, y)

# Plot Cost Function
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.title('Cost Function vs. Iterations')
plt.show()

# Cross-validation score
cv_accuracy = cross_validate(X, y, model)
print(f'Cross-Validation Accuracy: {cv_accuracy:.2f}')

# Train Scikit-Learn Model for Comparison
clf = LogisticRegression(solver='lbfgs', max_iter=1000)
clf.fit(X, y)
print(f'Sklearn Coefficients: {clf.coef_}')
print(f'Sklearn Intercept: {clf.intercept_}')
