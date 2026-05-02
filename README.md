# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize weight, bias, learning rate, number of iterations, and input data.

2. Repeatedly compute predictions using current parameters and the sigmoid function.

3. Compare predictions with actual values and adjust weight and bias using gradient descent.

4. After training, use the final parameters to generate predictions and plot the results.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: KARTHIKEYAN A
RegisterNumber:  212225230131
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Load and Preprocess Data
data = pd.read_csv("Placement_Data.csv")
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

# Selecting more features to improve model performance
features = ['ssc_p', 'hsc_p', 'degree_p', 'mba_p']
X = data[features].values
y = data['status'].values

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Adding Bias term (column of 1s)
m = len(y)
X = np.c_[np.ones(m), X]

# 2. Define Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500))) # Clip to prevent overflow

def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    # Adding epsilon to log to prevent log(0) errors
    epsilon = 1e-15
    return (-1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

# 3. Training Configuration
theta = np.zeros(X.shape[1])
alpha = 0.05    # Learning rate
iterations = 1000
cost_history = []

# 4. Gradient Descent Loop
for i in range(iterations):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    cost_history.append(cost_function(X, y, theta))

# 5. Evaluation
y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)
accuracy = np.mean(y_pred == y) * 100

print("Final Weights:", theta)
print(f"Accuracy: {accuracy:.2f}%")



# 6. Visualization
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color='blue', linewidth=2)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Convergence of Gradient Descent")
plt.grid(True)
plt.show()
```
## Output:
<img width="1174" height="597" alt="image" src="https://github.com/user-attachments/assets/856fcaaf-0351-4e5b-b560-ee32af8a2201" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

