import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
  
# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data (as pandas dataframes) 
X = cdc_diabetes_health_indicators.data.features 
y = cdc_diabetes_health_indicators.data.targets 
  
# X.to_csv('data2x.csv',index =False)
# y.to_csv('data2y.csv', index = False)

X = X.values
y = y.values
y = y.reshape(-1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, batch_size=32):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.coefficients = None
        self.intercept = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.coefficients = np.zeros(n)
        self.intercept = 0

        for _ in range(self.num_iterations):
            for i in range(0, m, self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]

                z = X_batch.dot(self.coefficients) + self.intercept
                h = self.sigmoid(z)
                gradient = X_batch.T.dot(h - y_batch) / len(X_batch)
                
                self.coefficients -= self.learning_rate * gradient
                self.intercept -= self.learning_rate * np.mean(h - y_batch)

    def predict(self, X):
        z = X.dot(self.coefficients) + self.intercept
        return (self.sigmoid(z) >= 0.5).astype(int)
    
def cost_fn(x, y, w):
        N, D = x.shape
        x = np.column_stack([x,np.ones(N)])                                          
        z = np.dot(x, w)
        J = np.mean(y * np.log1p(np.exp(-z)) + (1-y) * np.log1p(np.exp(z)))  #log1p calculates log(1+x) to remove floating point inaccuracies 
        return J

model = LogisticRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Experiment 1: Logistic Regression Performance")
print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Test Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Test Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"Test F1-score: {f1_score(y_test, y_test_pred):.4f}")

# # Experiment 2: Report weights of features
# feature_importance = pd.DataFrame({
#     'Feature': X.columns,
#     'Coefficient': model.weights
# })
# feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
# feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

# print("\nExperiment 2: Top 10 Feature Weights")
# print(feature_importance.head(10))

# # Experiment 3: Sample growing subsets of training data
# train_sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# train_scores = []
# test_scores = []

# for size in train_sizes:
#     X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
#     model = LogisticRegression()
#     model.fit(X_train_subset, y_train_subset)
    
#     train_pred = model.predict(X_train_subset)
#     test_pred = model.predict(X_test)
    
#     train_scores.append(accuracy_score(y_train_subset, train_pred))
#     test_scores.append(accuracy_score(y_test, test_pred))

# plt.figure(figsize=(10, 6))
# plt.plot(train_sizes, train_scores, label='Train Accuracy')
# plt.plot(train_sizes, test_scores, label='Test Accuracy')
# plt.xlabel('Training Set Size')
# plt.ylabel('Accuracy Score')
# plt.title('Learning Curve')
# plt.legend()
# plt.show()

# # Experiment 4: Try different mini-batch sizes
# batch_sizes = [8, 16, 32, 64, 128]
# batch_scores = []

# for batch_size in batch_sizes:
#     model = LogisticRegression(batch_size=batch_size)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     batch_scores.append(accuracy_score(y_test, y_pred))

# plt.figure(figsize=(10, 6))
# plt.plot(batch_sizes, batch_scores, marker='o')
# plt.xlabel('Batch Size')
# plt.ylabel('Test Accuracy')
# plt.title('Effect of Batch Size on Model Performance')
# plt.xscale('log')
# plt.show()

# print("\nExperiment 4: Mini-batch Size Performance")
# for batch_size, score in zip(batch_sizes, batch_scores):
#     print(f"Batch size {batch_size}: Test Accuracy = {score:.4f}")

# # Experiment 5: Try different learning rates
# learning_rates = [0.001, 0.01, 0.1]
# lr_scores = []

# for lr in learning_rates:
#     model = LogisticRegression(learning_rate=lr)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     lr_scores.append(accuracy_score(y_test, y_pred))

# plt.figure(figsize=(10, 6))
# plt.plot(learning_rates, lr_scores, marker='o')
# plt.xlabel('Learning Rate')
# plt.ylabel('Test Accuracy')
# plt.title('Effect of Learning Rate on Model Performance')
# plt.xscale('log')
# plt.show()

# print("\nExperiment 5: Learning Rate Performance")
# for lr, score in zip(learning_rates, lr_scores):
#     print(f"Learning rate {lr}: Test Accuracy = {score:.4f}")