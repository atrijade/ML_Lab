import numpy as np
import matplotlib.pyplot as plt

def lwr(x, y, query_x, tau=0.5):
    X = np.c_[np.ones(len(x)), x]
    W = np.diag(np.exp(-((x - query_x)**2) / (2 * tau**2)))
    theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
    return np.array([1, query_x]) @ theta

np.random.seed(0)
x = np.linspace(0, 10, 100)
y = np.sin(x) + 0.3 * np.random.randn(100)

xq = np.linspace(0, 10, 300)
yp = np.array([lwr(x, y, xi, tau=0.3) for xi in xq])

plt.scatter(x, y, color='lightblue', label='Data')
plt.plot(xq, yp, color='red', label='LWR')
plt.title("Locally Weighted Regression")

plt.show()