from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# fetch dataset 
infrared_thermography_temperature = fetch_ucirepo(id=925) 
  
# data (as pandas dataframes) 
X = infrared_thermography_temperature.data.features 
y = infrared_thermography_temperature.data.targets 
  
# metadata 
print(infrared_thermography_temperature.metadata) 
  
# variable information 
print(infrared_thermography_temperature.variables) 



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class LinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        X = np.column_stack((np.ones(X.shape[0]), X)) # Add a column of ones to X for the intercept term
        self.theta = np.linalg.inv(X.T @ X) @ X.T @ y # Compute the analytical solution

    def predict(self, X):
        X = np.column_stack((np.ones(X.shape[0]), X)) # Add a column of ones to X for the intercept term
        return X @ self.theta


# Train linear regression using analytical solution
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predictions on training and test sets
y_train_pred = lin_reg.predict(X_train)
y_test_pred = lin_reg.predict(X_test)

# Evaluate using Mean Squared Error (MSE)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Linear Regression - Train MSE: {train_mse}")
print(f"Linear Regression - Test MSE: {test_mse}")


