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

# Handle missing values
nan_rows = X.isnull().any(axis=1)
X_clean = X[~nan_rows]
y_clean = y[~nan_rows]

# Handle categorical features
categorical_columns = ['Age', 'Gender', 'Ethnicity']
X_dummies = pd.get_dummies(X_clean, columns=categorical_columns, drop_first=True)

# Convert boolean columns to integer
bool_columns = X_dummies.select_dtypes(include=['bool']).columns
for col in bool_columns:
    X_dummies[col] = X_dummies[col].astype(int)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dummies)

# Select the target variable
y_final = y_clean['aveOralM']

# X_scaled and y_final are now ready for model training

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_final, test_size=0.2, random_state=42)

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
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
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
    'Feature': X_dummies.columns,
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
    x_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
    model = LinearRegression()
    model.fit(x_train_subset, y_train_subset)
    
    train_pred = model.predict(x_train_subset)
    test_pred = model.predict(X_test)
    
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
# The SGDLinearRegression class is assumed to be defined as provided

def minibatch_iteration(X, y, max_iters=200):
    N = X.shape[0]
    batch_size = 2**3
    loss_histories = []

    # Mini-batch iterations
    for _ in range(5):
        model = SGDLinearRegression()
        losses = []
        for iteration in range(max_iters):
            model.fit(X, y, max_iterations=1, learning_rate=5e-2, batch_size=batch_size)
            _, loss = model.linear_loss(np.column_stack([np.ones(N), X]), y)
            losses.append(loss)
        loss_histories.append(losses)
        batch_size *= 2

    # Fully batched baseline (batch_size = N)
    model_fully_batched = SGDLinearRegression()
    full_batch_losses = []
    for iteration in range(max_iters):
        model_fully_batched.fit(X, y, max_iterations=1, learning_rate=5e-2, batch_size=N)
        _, loss = model_fully_batched.linear_loss(np.column_stack([np.ones(N), X]), y)
        full_batch_losses.append(loss)
    
    loss_histories.append(full_batch_losses)
    
    return loss_histories

# Running the experiment
loss_histories = minibatch_iteration(X_train, y_train)

# Adding the fully batched baseline
batch_sizes = [2**i for i in range(3, 8)] + [X_train.shape[0]]  # List of batch sizes + full batch

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(12, 6))
k = 0

for i in range(3):
    for j in range(2):
        if k < len(loss_histories):
            loss_history = np.array(loss_histories[k]).reshape(-1, 20).mean(1)
            axes[i, j].plot(loss_history)
            if batch_sizes[k] == X_train.shape[0]:
                title = f"Fully Batched (Batch Size: {batch_sizes[k]})"
            else:
                title = f"Batch Size: {batch_sizes[k]}"
            axes[i, j].set(xlabel="Iterations (x20)", ylabel="Average Loss",
                           title=title)
            axes[i, j].set_yscale('log')
            k += 1
        else:
            fig.delaxes(axes[i, j])

fig.tight_layout()
plt.show()

# Print summary statistics
print("Summary of Mini-batch Size Influence on SGD Linear Regression Loss:")
for batch_size, losses in zip(batch_sizes, loss_histories):
    print(f"Batch size: {batch_size}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Mean loss: {np.mean(losses):.6f}")
    print(f"  Min loss: {np.min(losses):.6f}")
    print()

# Experiment 5: Try different learning rates
def learning_rate_experiment(X, y, learning_rates, max_iters=200, batch_size=32):
    loss_histories = []

    for lr in learning_rates:
        model = SGDLinearRegression()
        losses = []

        for iteration in range(max_iters):
            model.fit(X, y, max_iterations=1, learning_rate=lr, batch_size=batch_size)
            _, loss = model.linear_loss(np.column_stack([np.ones(X.shape[0]), X]), y)
            losses.append(loss)

        loss_histories.append(losses)

    return loss_histories

# Experiment parameters
learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
max_iterations = 1500
batch_size = 32

# Run the experiment
loss_histories = learning_rate_experiment(X_train, y_train, learning_rates, max_iterations, batch_size)

# Plotting
fig, axs = plt.subplots(1, len(learning_rates), figsize=(15, 5))

for i, learning_rate in enumerate(learning_rates):
    axs[i].plot(np.array(loss_history).reshape(-1, 200).mean(1))
    axs[i].set_title(f"Learning Rate = {learning_rate}")
    axs[i].set_xlabel("Iteration")
    axs[i].set_ylabel("Training loss")
    axs[i].grid(linestyle='--', linewidth=0.5)

plt.show()

# Print summary statistics
print("\nSummary of Learning Rate Influence on SGD Linear Regression Loss:")
for lr, losses in zip(learning_rates, loss_histories):
    print(f"Learning Rate: {lr}")
    print(f"  Min loss: {np.min(losses):.6f}")
    print()

# Experiment 6: Compare analytical solution with mini-batch SGD

def cross_validation_experiment(X_train, y_train, X_test, y_test):
    num_folds = 5
    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)
    results = {}
    best_val = np.inf
    best_lr = None
    best_model = None
    learning_rates = np.linspace(1e-2, 1e-5, 20)

    for lr in learning_rates:
        val_mse_avg = 0
        for i in range(num_folds):
            X_train_temp = np.concatenate([X_train_folds[j] for j in range(num_folds) if j != i])
            y_train_temp = np.concatenate([y_train_folds[j] for j in range(num_folds) if j != i])

            model = SGDLinearRegression()
            model.fit(X_train_temp, y_train_temp, max_iterations=500, learning_rate=lr, verbose=False)

            y_train_pred = model.predict(X_train_temp)
            y_val_pred = model.predict(X_train_folds[i])
            y_test_pred = model.predict(X_test)

            train_mse = np.mean((y_train_temp - y_train_pred) ** 2)
            val_mse = np.mean((y_train_folds[i] - y_val_pred) ** 2)
            test_mse = np.mean((y_test - y_test_pred) ** 2)

            results[(lr, i)] = (train_mse, val_mse, test_mse)
            val_mse_avg += val_mse

        val_mse_avg /= num_folds
        if val_mse_avg < best_val:
            best_val = val_mse_avg
            best_lr = lr
            best_model = model

    # Print out results
    for lr, i in sorted(results):
        train_mse, val_mse, test_mse = results[(lr, i)]
        print(f'lr {lr:.6e} train MSE: {train_mse:.6f} val MSE: {val_mse:.6f} test MSE: {test_mse:.6f}')

    print(f'Best average validation mean squared error achieved during cross-validation: {best_val:.6f}, with learning rate {best_lr:.6e}')

    return best_model, best_lr

# Assuming X_train, y_train, X_test, y_test are your data
best_model, best_lr = cross_validation_experiment(X_train, y_train, X_test, y_test)

# Use the best model to make predictions on the test set
y_test_pred = best_model.predict(X_test)
final_test_mse = np.mean((y_test - y_test_pred) ** 2)
print(f'Final Test MSE with best model: {final_test_mse:.6f}')
