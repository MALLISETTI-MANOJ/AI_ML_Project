import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================================
# Utility Functions
# ==========================================================

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def normalize(data):
    data = data.astype(float)
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


# ==========================================================
# K-Means function 
# ==========================================================

def initialize_centroids(data, k):
    np.random.seed(42)
    indices = np.random.choice(len(data), k, replace=False)
    return data[indices]


def assign_clusters(data, centroids):
    labels = []
    for point in data:
        distances = [euclidean_distance(point, c) for c in centroids]
        labels.append(np.argmin(distances))
    return np.array(labels)


def update_centroids(data, labels, k):
    centroids = []
    for i in range(k):
        points = data[labels == i]
        if len(points) == 0:
            centroids.append(data[np.random.randint(len(data))])
        else:
            centroids.append(np.mean(points, axis=0))
    return np.array(centroids)


def kmeans(data, k, max_iter=100):
    centroids = initialize_centroids(data, k)

    for _ in range(max_iter):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels


# ==========================================================
# Davies-Bouldin function
# ==========================================================

def davies_bouldin(data, labels, centroids):
    k = len(centroids)
    sigmas = []

    for i in range(k):
        cluster_points = data[labels == i]
        if len(cluster_points) == 0:
            sigmas.append(0)
        else:
            sigmas.append(np.mean([
                euclidean_distance(p, centroids[i]) for p in cluster_points
            ]))

    db_values = []

    for i in range(k):
        ratios = []
        for j in range(k):
            if i != j:
                numerator = sigmas[i] + sigmas[j]
                denominator = euclidean_distance(centroids[i], centroids[j])
                ratios.append(numerator / denominator)
        db_values.append(max(ratios))

    return np.mean(db_values)


# ==========================================================
# Fuzzy C-Means function
# ==========================================================

def fuzzy_c_means(data, k, m=2, max_iter=100):
    n = len(data)
    np.random.seed(42)
    U = np.random.dirichlet(np.ones(k), size=n)

    for _ in range(max_iter):
        centroids = []
        for j in range(k):
            numerator = np.sum((U[:, j] ** m).reshape(-1, 1) * data, axis=0)
            denominator = np.sum(U[:, j] ** m)
            centroids.append(numerator / denominator)
        centroids = np.array(centroids)

        for i in range(n):
            for j in range(k):
                denom = 0
                for c in range(k):
                    dist1 = euclidean_distance(data[i], centroids[j])
                    dist2 = euclidean_distance(data[i], centroids[c])
                    if dist2 == 0:
                        dist2 = 1e-10
                    ratio = dist1 / dist2
                    denom += ratio ** (2 / (m - 1))
                U[i, j] = 1 / denom

    return centroids, U


# ==========================================================
# Plotting Function (3D)
# ==========================================================

def plot_3d(data, labels, centroids, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2],
               c=labels, cmap='viridis', s=20)

    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
               marker='X', s=200, c='red')

    ax.set_title(title)
    plt.show()


# ==========================================================
# Load Dataset Columns
# ==========================================================

data = pd.read_csv("drug_consumption.data", header=None)

Age = data[1].values.reshape(-1, 1)
Neuroticism = data[6].values.reshape(-1, 1)
Agreeableness = data[9].values.reshape(-1, 1)
Conscientiousness = data[10].values.reshape(-1, 1)
Impulsivity = data[11].values.reshape(-1, 1)
Sensation = data[12].values.reshape(-1, 1)

# ==========================================================
# QUESTION 1a
# ==========================================================

print("\n========== QUESTION 1a ==========")

features_3 = normalize(np.hstack((Age, Impulsivity, Sensation)))

centroids, labels = kmeans(features_3, 2)

print("Centroids for k=2:")
print(centroids)

plot_3d(features_3, labels, centroids, "Q1a: k=2")

# =======================================================
# QUESTION 1b
# =======================================================

print("\n======== QUESTION 1b =========")

for k in range(1, 9):
    centroids, labels = kmeans(features_3, k)
    print(f"\nk = {k}")
    print("Centroids:")
    print(centroids)

    if k >= 2:
        plot_3d(features_3, labels, centroids, f"Q1b: k={k}")

# =========================================================
# QUESTION 1c
# =========================================================

print("\n========== QUESTION 1c ==========")

best_k = None
best_db = float('inf')

for k in range(2, 9):
    centroids, labels = kmeans(features_3, k)
    db = davies_bouldin(features_3, labels, centroids)
    print(f"k = {k} DB = {db}")

    if db < best_db:
        best_db = db
        best_k = k

print("\nBest k based on DB:", best_k)

# =======================================================
# QUESTION 2a
# =======================================================

print("\n========== QUESTION 2a ==========")

features_4 = normalize(np.hstack((features_3, Neuroticism)))

for k in range(2, 9):
    centroids, labels = kmeans(features_4, k)
    db = davies_bouldin(features_4, labels, centroids)
    print(f"k = {k} DB = {db}")

# ======================================================
# QUESTION 2b
# =======================================================

print("\n========== QUESTION 2b ==========")

features_5 = normalize(np.hstack((features_4, Conscientiousness)))

for k in range(2, 9):
    centroids_5, labels = kmeans(features_5, k)
    db = davies_bouldin(features_5, labels, centroids_5)
    print(f"k = {k} DB = {db}")


# =======================================================
# QUESTION 3a
# =======================================================

print("\n========== QUESTION 3a ==========")

# Using the best k obtained from Question 1c
print(f"Using best k from Q1c: k = {best_k}")

# ----- K-MEANS -----
centroids_k, labels_k = kmeans(features_5, best_k)

print("\nK-means Centroids:")
print(centroids_k)

# Since features_5 has 5 dimensions, we visualize using first 3 features
plot_3d(features_5[:, :3], labels_k, centroids_k[:, :3],
        "Q3a: K-means Clustering (First 3 Features)")


# ====== FUZZY C-MEANS ====
centroids_f, membership = fuzzy_c_means(features_5, best_k)

print("\nFuzzy C-Means Centroids:")
print(centroids_f)

# Convert soft membership to hard labels for visualization
hard_labels = np.argmax(membership, axis=1)

plot_3d(features_5[:, :3], hard_labels, centroids_f[:, :3],
        "Q3a: Fuzzy C-Means Clustering (First 3 Features)")


# ========================================================
# QUESTION 3b
# =========================================================

print("\n========== QUESTION 3b ==========")

# Compute DB index for both clustering methods
db_k = davies_bouldin(features_5, labels_k, centroids_k)
db_f = davies_bouldin(features_5, hard_labels, centroids_f)

print("Davies-Bouldin Index Comparison:")
print("K-means DB:", db_k)
print("Fuzzy C-Means DB:", db_f)

if db_k < db_f:
    print("\n K-means produces more compact and well-separated clusters.")
else:
    print("\n Fuzzy C-Means produces better clustering structure.")


# =========================================================
# QUESTION 3c
# ==========================================================

print("\n========== QUESTION 3c ==========")

features_6 = normalize(np.hstack((features_5, Agreeableness)))

centroids_6, labels_6 = kmeans(features_6, best_k)

print("Centroids with sixth feature:")
print(centroids_6)

db_6 = davies_bouldin(features_6, labels_6, centroids_6)

print("\nDB after adding sixth feature:", db_6)
