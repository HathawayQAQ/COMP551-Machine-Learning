from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
  
# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data (as pandas dataframes) 
X = cdc_diabetes_health_indicators.data.features 
y = cdc_diabetes_health_indicators.data.targets 
  
# metadata 
print(cdc_diabetes_health_indicators.metadata) 
  
# variable information 
print(cdc_diabetes_health_indicators.variables) 

class LogisticRegression:
    def __init__(self):
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, learning_rate=0.01, num_iterations=1000):
        X = np.column_stack((np.ones(X.shape[0]), X)) # Add a column of ones to X for the intercept term
        self.theta = np.zeros(X.shape[1]) # Initialize theta

        for _ in range(num_iterations):
            z = X @ self.theta
            h = self.sigmoid(z)
            gradient = X.T @ (h - y) / y.size
            self.theta -= learning_rate * gradient

    def predict(self, X):
        X = np.column_stack((np.ones(X.shape[0]), X)) # Add a column of ones to X for the intercept term
        return self.sigmoid(X @ self.theta)

# X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
# y_train = np.array([0, 0, 1, 1])
# model = LogisticRegression()
# model.fit(X_train, y_train)
# X_test = np.array([[3, 3], [5, 5]])
# predictions = model.predict(X_test)
# print(predictions)


class MiniBatchSGD:
    def __init__(self, model_type='linear'):
        self.theta = None
        self.model_type = model_type

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_gradient(self, X, y, y_pred):
        if self.model_type == 'linear':
            return X.T @ (y_pred - y) / len(y)
        elif self.model_type == 'logistic':
            return X.T @ (self.sigmoid(y_pred) - y) / len(y)
        else:
            raise ValueError("Invalid model type. Choose 'linear' or 'logistic'.")

    def fit(self, X, y, learning_rate=0.01, num_iterations=1000, batch_size=32):
        # Add a column of ones to X for the intercept term
        X = np.column_stack((np.ones(X.shape[0]), X))
        
        # Initialize theta
        self.theta = np.zeros(X.shape[1])

        n_samples = X.shape[0]
        
        for _ in range(num_iterations):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch loop
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                y_pred = X_batch @ self.theta
                gradient = self.compute_gradient(X_batch, y_batch, y_pred)
                self.theta -= learning_rate * gradient

    def predict(self, X):
        # Add a column of ones to X for the intercept term
        X = np.column_stack((np.ones(X.shape[0]), X))
        
        if self.model_type == 'linear':
            return X @ self.theta
        elif self.model_type == 'logistic':
            return self.sigmoid(X @ self.theta)

# Example usage for linear regression:
# X_train_linear = np.array([[1], [2], [3], [4], [5]])
# y_train_linear = np.array([2, 4, 5, 4, 5])
# 
# model_linear = MiniBatchSGD(model_type='linear')
# model_linear.fit(X_train_linear, y_train_linear, batch_size=2)
# 
# X_test_linear = np.array([[6], [7]])
# predictions_linear = model_linear.predict(X_test_linear)
# print("Linear Regression Predictions:", predictions_linear)

# Example usage for logistic regression:
# X_train_logistic = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
# y_train_logistic = np.array([0, 0, 1, 1])
# 
# model_logistic = MiniBatchSGD(model_type='logistic')
# model_logistic.fit(X_train_logistic, y_train_logistic, batch_size=2)
# 
# X_test_logistic = np.array([[3, 3], [5, 5]])
# predictions_logistic = model_logistic.predict(X_test_logistic)
# print("Logistic Regression Predictions:", predictions_logistic)