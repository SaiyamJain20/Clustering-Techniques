import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import pandas as pd

# Parameters
epsilon = 8
min_pts = 3
letter_width = 10
letter_height = 15
letter_spacing = 10
letters = ['S', 'A', 'I']

#letters
letter_shapes = {
    'S': np.array([
        [0.8, 1.0], [0.5, 0.8], [0.2, 0.6],
        [0.5, 0.5], [0.8, 0.4], [0.5, 0.2], [0.2, 0.0],
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
orig_letters = []

for i, letter in enumerate(letters):
    offset_x = i * (letter_width + letter_spacing)
    scaled_points = letter_shapes[letter] * [letter_width, letter_height]
    translated_points = scaled_points + [offset_x, 0]
    all_points.append(translated_points)
    all_labels.extend([f"{letter}{j+1}" for j in range(7)])
    orig_letters.extend([letter] * 7)

points = np.vstack(all_points)
n_points = len(points)

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

def print_dbscan_distances():
    table = PrettyTable()
    headers = ["Point"] + all_labels
    table.field_names = headers
    for i in range(n_points):
        row = [all_labels[i]]
        for j in range(n_points):
            d = np.linalg.norm(points[i] - points[j])
            row.append(f"{d:.2f}*" if d <= epsilon and i != j else f"{d:.2f}" if i != j else "0.00")
        table.add_row(row)
    print("\n=== Distance Table (\u03B5 = {:.2f}) ===".format(epsilon))
    print(table)

print_dbscan_distances()

# Neighborhood and classification
neighborhoods = [np.where(np.linalg.norm(points - points[i], axis=1) <= epsilon)[0] for i in range(n_points)]
point_types = ['noise'] * n_points
for i, neighs in enumerate(neighborhoods):
    if len(neighs) >= min_pts:
        point_types[i] = 'core'
for i in range(n_points):
    if point_types[i] == 'noise':
        for j in neighborhoods[i]:
            if point_types[j] == 'core':
                point_types[i] = 'border'
                break

# DBSCAN clustering
cluster_id = 0
assignments = [-1] * n_points
visited = set()
for i in range(n_points):
    if point_types[i] != 'core' or i in visited:
        continue
    seeds = [i]
    while seeds:
        curr = seeds.pop()
        if curr in visited:
            continue
        visited.add(curr)
        assignments[curr] = cluster_id
        if point_types[curr] == 'core':
            for neighbor in neighborhoods[curr]:
                if assignments[neighbor] == -1:
                    seeds.append(neighbor)
    cluster_id += 1

point_map = {i: all_labels[i] for i in range(n_points)}

# Cluster distribution analysis
data = pd.DataFrame({
    'No': range(0, n_points),
    'Point': all_labels,
    'Original_Letter': orig_letters,
    'Assigned_Cluster': [f"Cluster {c}" if c != -1 else 'Noise' for c in assignments],
    'Neighbours': [(len(neigh) - 1) for neigh in neighborhoods],
    'Neighbour_List': [', '.join([point_map[j] for j in neigh if j != i]) for i, neigh in enumerate(neighborhoods)],
    'Point_Type': point_types
})
print("\n=== Cluster Distribution Analysis ===")
dist_table = PrettyTable()
dist_table.field_names = list(data.columns)
for _, row in data.iterrows():
    dist_table.add_row(row.tolist())
print(dist_table)

# Letter-wise breakdown
for letter in letters:
    subset = data[data['Original_Letter'] == letter]
    print(f"\nLetter {letter} cluster distribution:")
    print(subset['Assigned_Cluster'].value_counts())

# Accuracy analysis
non_noise = data[data['Assigned_Cluster'] != 'Noise']
if not non_noise.empty:
    cluster_to_letter = {}
    for cluster in non_noise['Assigned_Cluster'].unique():
        counts = non_noise[non_noise['Assigned_Cluster'] == cluster]['Original_Letter'].value_counts()
        cluster_to_letter[cluster] = counts.idxmax()
    print("\nCluster to letter mapping:")
    print(cluster_to_letter)

    correct = sum(
        cluster_to_letter[row['Assigned_Cluster']] == row['Original_Letter']
        for _, row in non_noise.iterrows()
    )
    print("\nDBSCAN Accuracy (excluding noise): {:.2%}".format(correct / len(non_noise)))
    correct_total = sum(
        row['Assigned_Cluster'] != 'Noise' and \
        cluster_to_letter.get(row['Assigned_Cluster'], '') == row['Original_Letter']
        for _, row in data.iterrows()
    )
    print("DBSCAN Accuracy (including noise): {:.2%}".format(correct_total / len(data)))

# Visualization
def plot_dbscan():
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
    markers = {'core': 'o', 'border': 's', 'noise': 'x'}
    plt.figure(figsize=(10, 6))
    for i in range(n_points):
        c = assignments[i]
        color = 'gray' if c == -1 else colors[c % len(colors)]
        plt.scatter(points[i][0], points[i][1], c=color, s=100,
                    marker=markers[point_types[i]])
        plt.text(points[i][0]+0.3, points[i][1]+0.3, all_labels[i], fontsize=9)
    plt.title("DBSCAN Clustering of S, A, I")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.show()

plot_dbscan()
