import numpy as np
import matplotlib.pyplot as plt
import random as rand
from prettytable import PrettyTable
import pandas as pd

np.set_printoptions(precision=2, suppress=True)
k = 3  # number of clusters
max_iters = 10

letter_width = 10
letter_height = 15
letter_spacing = 10
letters = ['S', 'A', 'I']

# letters
letter_shapes = {
    'S': np.array([
        [1.0, 1.0], [0.5, 0.8], [0.0, 0.6],
        [0.5, 0.5], [1.0, 0.4], [0.5, 0.2], [0.0, 0.0],
    ]),
    'A': np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, 1.0],
        [0.1, 0.5], [0.9, 0.5], [0.3, 0.5], [0.6, 0.5],
    ]),
    'I': np.array([
        [0.5, 1.0], [0.0, 0.0], [0.5, 0.5],
        [1.0, 0.0], [0.5, 0.0], [0.0, 1.0], [1.0, 1.0],
    ])
}

all_points = []
all_labels = []

for i, letter in enumerate(letters):
    offset_x = i * (letter_width + letter_spacing)
    scaled_points = letter_shapes[letter] * [letter_width, letter_height]
    translated_points = scaled_points + [offset_x, 0]
    all_points.append(translated_points)
    all_labels.extend([f"{letter}{j+1}" for j in range(7)])

points = np.vstack(all_points)
n_points = points.shape[0]

# Randomly pick 3 points as initial centroids (or choose fixed for repeatability)
initial_indices = []
for _ in range(k):
    while True:
        idx = rand.randint(0, n_points - 1)
        if idx not in initial_indices:
            initial_indices.append(idx)
            break
centroids = points[initial_indices].copy()

def printDistance(centroids, points, assignments):
    table = PrettyTable()
    centroid_headers = [f"Dist to C{i}" for i in range(len(centroids))]
    table.field_names = ["Point #", "X", "Y"] + centroid_headers + ["Assigned Cluster"]

    distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)

    for idx, point in enumerate(points):
        row = [all_labels[idx], round(point[0], 2), round(point[1], 2)]
        dists = distances[idx]
        row += [f"{d:.2f}" for d in dists]
        row.append(f"{assignments[idx]}")
        table.add_row(row)

    print(table)

def plot(points, assignments, centroids, iteration):
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'gray']
    plt.figure(figsize=(8, 6))
    
    for i in range(k):
        cluster_points = points[assignments == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i}')
        plt.scatter(*centroids[i], color=colors[i], edgecolors='black', marker='X', s=200, linewidths=2)
        plt.text(centroids[i][0] + 0.5, centroids[i][1] + 0.5, f'C{i}', fontsize=12, weight='bold')
    
    # label (e.g., S1, A1, I1, etc.)
    for idx, point in enumerate(points):
        plt.text(point[0] + 0.2, point[1] + 0.2, all_labels[idx], fontsize=9, weight='bold')

    plt.title(f"K-Means Iteration {iteration}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()


def printClusters(points, assignments):
    print("\nCluster Assignments:")
    for i in range(k):
        cluster_points = points[assignments == i]
        cluster_labels = [all_labels[idx] for idx in range(len(points)) if assignments[idx] == i]
        print(f"\nCluster {i} ({len(cluster_points)} points):")
        print(f"  {cluster_labels}")
    print()


# Base Letters Plot
plt.figure(figsize=(10, 6))
plt.scatter(points[:, 0], points[:, 1], color='blue', s=60)

for (x, y), label in zip(points, all_labels):
    plt.text(x + 0.3, y + 0.3, label, fontsize=12, color='black')

plt.title("Letter Shape Representation: S, A, I")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Iterations (max_iters = 10)
for iteration in range(1, max_iters + 1):
    print(f"\n====== Iteration {iteration} ======")
    print("Current Centroids:")
    for i, c in enumerate(centroids):
        print(f"  Centroid {i}: {c.round(2)}")

    # Step 1: Assign points to nearest centroid
    distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)
    assignments = np.argmin(distances, axis=1)

    printDistance(centroids, points, assignments)
    printClusters(points, assignments)

    # Step 2: Recompute centroids
    new_centroids = np.zeros_like(centroids)
    for i in range(k):
        cluster_points = points[assignments == i]
        if len(cluster_points) > 0:
            new_centroids[i] = cluster_points.mean(axis=0)
        else:
            new_centroids[i] = centroids[i] 

    print("New Centroids:")
    for i, c in enumerate(new_centroids):
        print(f"  Centroid {i}: {c.round(2)}")

    plot(points, assignments, centroids, iteration)
    if np.allclose(centroids, new_centroids):
        print("Centroids have stabilized. Clustering complete.")
        break

    centroids = new_centroids


# Post-Clustering Analysis
cluster_df = pd.DataFrame({
    'Point': all_labels,
    'Original_Letter': [label[0] for label in all_labels],
    'Assigned_Cluster': [f'Cluster {i}' for i in assignments]
})

print("\nClustering Analysis:")
print(cluster_df)

# Distribution per letter
for letter in letters:
    print(f"\nLetter {letter} distribution across clusters:")
    print(cluster_df[cluster_df['Original_Letter'] == letter]['Assigned_Cluster'].value_counts())

# Cluster to Letter mapping
cluster_to_letter = {}
for i in range(k):
    letter_counts = []
    for letter in letters:
        count = sum((cluster_df['Original_Letter'] == letter) & 
                    (cluster_df['Assigned_Cluster'] == f'Cluster {i}'))
        letter_counts.append((letter, count))
    letter_counts.sort(key=lambda x: x[1], reverse=True)
    cluster_to_letter[f'Cluster {i}'] = letter_counts[0][0]

print("\nCluster to Letter mapping:")
print(cluster_to_letter)

# Accuracy calculation
correct = 0
for _, row in cluster_df.iterrows():
    if cluster_to_letter[row['Assigned_Cluster']] == row['Original_Letter']:
        correct += 1

accuracy = correct / len(cluster_df)
print(f"\nK-means clustering accuracy: {accuracy:.2%}")
