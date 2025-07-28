import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.cluster import KMeans

# Generate synthetic data (or replace with CSV loading below)
np.random.seed(110)
red = np.random.normal(3, 0.8, 40)
blue = np.random.normal(7, 1, 40)
data = np.sort(np.concatenate((red, blue))).reshape(-1, 1)  # Reshape for kMeans

# Uncomment this to load data from CSV
# import pandas as pd
# data = pd.read_csv('yourfile.csv').values.reshape(-1, 1)
# Initial guesses
mean1, std1 = 2.1, 1.5
mean2, std2 = 6.0, 0.8

def estimate_mean(x, w):
    return np.sum(x * w) / np.sum(w)

def estimate_std(x, w, mean):
    return np.sqrt(np.sum(w * (x - mean) ** 2) / np.sum(w))

# EM iterations
for _ in range(10):
    p1 = norm.pdf(data, mean1, std1)
    p2 = norm.pdf(data, mean2, std2)
    total = p1 + p2
    w1 = p1 / total
    w2 = p2 / total

    mean1 = estimate_mean(data, w1)
    mean2 = estimate_mean(data, w2)
    std1 = estimate_std(data, w1, mean1)
    std2 = estimate_std(data, w2, mean2)

print(f"EM Means: {mean1:.2f}, {mean2:.2f}")
print(f"EM Stds : {std1:.2f}, {std2:.2f}")
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
kmeans_labels = kmeans.labels_
print("K-Means Centers:", kmeans.cluster_centers_.flatten())
x = np.linspace(min(data)-1, max(data)+1, 300)
plt.plot(x, norm.pdf(x, mean1, std1), label='EM Cluster 1')
plt.plot(x, norm.pdf(x, mean2, std2), label='EM Cluster 2')

# KMeans clusters
plt.scatter(data, np.zeros_like(data), c=kmeans_labels, cmap='bwr', marker='o', label='KMeans Data')

plt.legend()
plt.title("EM vs K-Means Clustering")
plt.show()
