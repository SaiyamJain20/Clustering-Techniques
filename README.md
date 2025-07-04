# Clustering Algorithms Implementation: K-Means & DBSCAN
This repository contains implementations of two popular clustering algorithms: K-Means and DBSCAN (Density-Based Spatial Clustering of Applications with Noise). Both implementations are demonstrated on a synthetic dataset consisting of 2D points representing the letters 'S', 'A', and 'I'.

## Project Structure
The repository contains the following files:

- KMeans.py: Implementation of the K-Means clustering algorithm
- DBScan.py: Implementation of the DBSCAN clustering algorithm

## Dataset
The dataset is synthetic and consists of 2D points that form the shapes of three letters: 'S', 'A', and 'I'. Each letter is represented by 7 data points. The points are positioned and labeled as follows:

- Letter 'S': Points S1-S7
- Letter 'A': Points A1-A7
- Letter 'I': Points I1-I7
The letters are placed with appropriate spacing to make the clustering task meaningful.

## Algorithms
### K-Means Implementation (KMeans.py)
K-Means is a partitioning clustering algorithm that divides the data into k clusters, where each point belongs to the cluster with the nearest mean (centroid).

#### Parameters:
- k = 3: Number of clusters
- max_iters = 10: Maximum number of iterations
#### Implementation Details:
##### Initialization: 
- The algorithm randomly selects k initial centroids from the dataset.
##### Iterative Process:
- Assignment Step: Each data point is assigned to the nearest centroid.
- Update Step: Centroids are recalculated as the mean of all points assigned to that cluster.
- Termination: The algorithm terminates when centroids stabilize or after reaching max_iters.
#### Visualization:
- The algorithm visualizes the clustering process at each iteration, showing:
- Points colored by cluster assignment
- Centroids marked with 'X'
- Point labels (e.g., S1, A1, I1)
#### Analysis:
- Displays distance tables at each iteration
- Shows cluster assignments and composition
- Calculates accuracy by mapping each cluster to the most frequent letter in that cluster

### DBSCAN Implementation (DBScan.py)
DBSCAN is a density-based clustering algorithm that groups together points that are close to each other and marks points in low-density regions as outliers (noise).

#### Parameters:
- epsilon = 8: Maximum distance between two points for them to be considered neighbors
- min_pts = 3: Minimum number of points required to form a dense region

#### Implementation Details:
##### Neighborhood Calculation: 
- For each point, the algorithm identifies all neighbors within distance epsilon.
##### Point Classification:
- Core Points: Points with at least min_pts neighbors
- Border Points: Points with fewer than min_pts neighbors but are in the neighborhood of a core point
- Noise Points: Points that are neither core nor border points
- Cluster Formation: Connected core points form clusters, with border points assigned to the clusters of their neighboring core points.

#### Visualization:
The algorithm visualizes the final clustering, showing:
- Points colored by cluster assignment (noise points in gray)
- Different markers for core, border, and noise points
- Point labels

#### Analysis:
- Displays a distance table showing pairwise distances between all points
- Shows cluster distribution and point classifications
- Calculates accuracy both including and excluding noise points

## Results and Comparison
Both algorithms attempt to cluster the points into their respective letters, but they have different approaches and results:

- K-Means: Partitions the data into exactly k clusters, even if some clusters should be merged or split.
- DBSCAN: Automatically determines the number of clusters based on density, and can identify noise points.

The accuracy metrics at the end of each implementation show how well the algorithms performed in identifying the original letter shapes.

## Requirements
The implementation requires the following Python libraries:

- NumPy
- Matplotlib
- Pandas
- PrettyTable
- Random (standard library)

## How to Run
To run the algorithms, simply execute the respective Python files:

- Each script will output detailed information about the clustering process and display visualizations at various stages.