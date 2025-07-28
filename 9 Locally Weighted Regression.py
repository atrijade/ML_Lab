import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gaussian kernel function
def kernel(x_point, x_mat, tau):
    m = x_mat.shape[0]
    weights = np.eye(m)
    for j in range(m):
        diff = x_point - x_mat[j]
        weights[j, j] = np.exp(-(diff @ diff.T) / (2 * tau ** 2))
    return weights

# Compute local weights and predict y for a point
def local_weight(x_point, x_mat, y_mat, tau):
    weights = kernel(x_point, x_mat, tau)
    theta = np.linalg.pinv(x_mat.T @ weights @ x_mat) @ x_mat.T @ weights @ y_mat
    return theta

# Predict y for all points
def local_weight_regression(x_mat, y_mat, tau):
    m = x_mat.shape[0]
    y_pred = np.zeros(m)
    for i in range(m):
        theta = local_weight(x_mat[i], x_mat, y_mat, tau)
        y_pred[i] = x_mat[i] @ theta
    return y_pred
# Load data from CSV file
data = pd.read_csv('LR.csv')  # Make sure it has 'colA' and 'colB' columns
x = np.array(data['colA'])
y = np.array(data['colB'])

# Prepare X matrix by adding bias term (1s column)
m = len(x)
X = np.column_stack((np.ones(m), x))
Y = y.reshape(-1, 1)  # convert to column vector
# Set tau (bandwidth parameter for locality)
tau = 0.5

# Predict using LWR
y_pred = local_weight_regression(X, Y, tau)

# Sort for plotting
sorted_indices = X[:, 1].argsort()
x_sorted = X[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

# Plot
plt.scatter(x, y, color='green', label='Original Data')
plt.plot(x_sorted[:, 1], y_pred_sorted, color='red', linewidth=2.5, label='LWR Curve')
plt.xlabel('colA')
plt.ylabel('colB')
plt.title('Locally Weighted Regression')
plt.legend()
plt.show()
