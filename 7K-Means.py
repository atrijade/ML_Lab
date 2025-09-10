# Import libraries
import pandas as pd
from sklearn.cluster import KMeans

# Step 1: Define the training data directly
data = pd.DataFrame({
    'X': [1, 1, 1, 10, 10, 10],
    'Y': [2, 4, 0, 2, 4, 0]
})

print("Data:")
print(data)

# Step 2: Initialize K-Means
# We want 2 clusters (k=2)
kmeans = KMeans(n_clusters=2, random_state=0)

# Step 3: Train the model
kmeans.fit(data)

# Step 4: Get cluster labels
labels = kmeans.labels_
print("\nCluster labels for each data point:")
print(labels)

# Step 5: Get cluster centers
centers = kmeans.cluster_centers_
print("\nCluster centers:")
print(centers)