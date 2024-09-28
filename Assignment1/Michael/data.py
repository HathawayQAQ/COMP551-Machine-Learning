import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Data acquisition
infrared_thermography_temperature = fetch_ucirepo(id=925)
X = infrared_thermography_temperature.data.features
y = infrared_thermography_temperature.data.targets

# One-hot encoding
x_dummies = pd.get_dummies(X, columns=['Age', 'Gender', 'Ethnicity'], drop_first=True)
bool_columns = x_dummies.select_dtypes(include=['bool']).columns
for col in bool_columns:
    x_dummies[col] = x_dummies[col].astype(int)

# Handling missing values
nan_rows = x_dummies.isnull().any(axis=1)
x_dummies = x_dummies[~nan_rows]

# Selecting the target variable
y = y['aveOralM']
y = y[~nan_rows]

# Feature scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_dummies)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        X = np.column_stack((np.ones(X.shape[0]), X))
        self.coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]
        return self

    def predict(self, X):
        return X.dot(self.coefficients) + self.intercept

# Mini-batch SGD Linear Regression
class SGDLinearRegression:
    def __init__(self, learning_rate=0.01, batch_size=32, epochs=100):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coefficients = np.zeros(n_features)
        self.intercept = 0
        
        for _ in range(self.epochs):
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]
                
                y_pred = X_batch.dot(self.coefficients) + self.intercept
                
                self.coefficients -= self.lr * (2/len(X_batch)) * X_batch.T.dot(y_pred - y_batch)
                self.intercept -= self.lr * (2/len(X_batch)) * np.sum(y_pred - y_batch)
        
        return self

    def predict(self, X):
        return X.dot(self.coefficients) + self.intercept

# Experiment 1: Report performance of linear regression
model = LinearRegression()
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
print(y_test_pred)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print("Experiment 1: Linear Regression Performance")
print(f"Training MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Training R-squared: {train_r2:.4f}")
print(f"Test R-squared: {test_r2:.4f}")
print(f"MAE (Train): {mae_train:.4f}")
print(f"MAE (Test): {mae_test:.4f}")


# model2 = LinearRegression()
# yh = model2.fit(x_test, y_test).predict(x_test)
# plt.plot(x_test, y_test, ".")
# plt.plot(x_test, yh, "g-", alpha=0.5)
# plt.xlabel("x")
# plt.ylabel("r")
# plt.show()

# Experiment 2: Report weights of features
feature_importance = pd.DataFrame({
    'Feature': x_dummies.columns,
    'Coefficient': model.coefficients
})
feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

print("\nExperiment 2: Top 10 Feature Weights")
print(feature_importance.head(10))

# Experiment 3: Sample growing subsets of training data
train_sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
train_scores = []
test_scores = []

for size in train_sizes:
    x_train_subset, _, y_train_subset, _ = train_test_split(x_train, y_train, train_size=size, random_state=42)
    model = LinearRegression()
    model.fit(x_train_subset, y_train_subset)
    
    train_pred = model.predict(x_train_subset)
    test_pred = model.predict(x_test)
    
    train_scores.append(r2_score(y_train_subset, train_pred))
    test_scores.append(r2_score(y_test, test_pred))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, label='Train R-squared')
plt.plot(train_sizes, test_scores, label='Test R-squared')
plt.xlabel('Training Set Size')
plt.ylabel('R-squared Score')
plt.title('Learning Curve')
plt.legend()
plt.show()

# Experiment 4: Try different mini-batch sizes
batch_sizes = [8, 16, 32, 64, 128]
batch_scores = []

def safe_r2_score(y_true, y_pred):
    try:
        return r2_score(y_true, y_pred)
    except ValueError:
        return float('-inf')
    
for batch_size in batch_sizes:
    model = SGDLinearRegression(batch_size=batch_size)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = safe_r2_score(y_test, y_pred)
    batch_scores.append(score)
    print(f"Batch size {batch_size}: Test R2= {score:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, batch_scores, marker='o')
plt.xlabel('Batch Size')
plt.ylabel('Test R2 Score')
plt.title('Effect of Batch Size on Model Performance')
plt.xscale('log')
plt.show()

print("\nExperiment 4: Mini-batch Size Performance")
for batch_size, score in zip(batch_sizes, batch_scores):
    print(f"Batch size {batch_size}: Test R2 = {score:.4f}")

# Experiment 5: Try different learning rates
learning_rates = [0.001, 0.01, 0.1]
lr_scores = []

for lr in learning_rates:
    model = SGDLinearRegression(learning_rate=lr)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = safe_r2_score(y_test, y_pred)
    lr_scores.append(score)
    print(f"Learning rate {lr}: Test R-squared = {score:.4f}")

print("\nExperiment 5: Learning Rate Performance")
for lr, score in zip(learning_rates, lr_scores):
    print(f"Learning rate {lr}: Test R-squared = {score:.4f}")

# Experiment 6: Compare analytical solution with mini-batch SGD
analytical_model = LinearRegression()
analytical_model.fit(x_train, y_train)
analytical_pred = analytical_model.predict(x_test)
analytical_score = r2_score(y_test, analytical_pred)

sgd_model = SGDLinearRegression()
sgd_model.fit(x_train, y_train)
sgd_pred = sgd_model.predict(x_test)
sgd_score = r2_score(y_test, sgd_pred)

print("\nExperiment 6: Analytical vs SGD Performance")
print(f"Analytical solution: Test R-squared = {analytical_score:.4f}")
print(f"Mini-batch SGD: Test R-squared = {sgd_score:.4f}")
