# K-Means and K-Medoids Demonstration

This repository demonstrates the K-Means and K-Medoids clustering algorithms and explains their concepts and workings.

## K-Means Clustering

K-Means is a popular partitioning-based clustering algorithm. It aims to find the K cluster centroids that minimize the total intraclass dispersion, which is the sum of squared distances between each data point and the nearest centroid.

The K-Means algorithm works as follows:

1. Initialize K random centroids: $z_{1},\dots,z_{K}$
2. Assign each data point $x^{(i)}$ to the nearest centroid $z_{j}$, creating $K$ clusters clusters $C_{1},\dots,C_{K}$.
3. Compute new centroids by calculating the mean of all data points in each cluster.
$$z_{j}=\frac{\displaystyle\sum_{i\in C_{j}}x^{(i)}}{|C_{j}|}$$
4. Repeat steps 2 and 3 until centroids stop changing or a maximum number of iterations is reached.

The K-Means algorithm tries to minimize the following objective function:

$$
J(c, \mu) = \sum_{i=1}^{K} \sum_{x \in c_i}{||x-\mu_i||^2}
$$

where $c_i$ is the $i\text{th}$ cluster, $\mu_i$ is the $i\text{th}$ centroid, and $x$ are the data points.

## K-Medoids Clustering

K-Medoids is another partitioning-based clustering algorithm, similar to K-Means but uses the medoids instead of centroids. Medoids are data points within the cluster that have the smallest average distance to all other points in the cluster.

The K-Medoids algorithm works as follows:

1. Initialize K random medoids: $z_{1},\dots,z_{K}$
2. Assign each data point to the nearest medoid, creating $K$ clusters $C_{1},\dots,C_{K}$.
3. Compute new medoids by selecting the data point $z_{j}$ in each cluster $C_{j}$ that has the smallest average distance to all other points $x^{(i)}$ in the cluster.
$$z_j = x^{(j)} \textbf{ s. t. } \displaystyle\sum_{x^{(i)} \in C_{j}}||x^{(j)} - x^{(i)}||^{2} \text{  is minimized, for  } \\{x^{(j)}|x^{(j)} \in C_{j}\\}$$ 
4. Repeat steps 2 and 3 until medoids stop changing or a maximum number of iterations is reached.

Compared to K-Means, K-Medoids is more robust to outliers and can handle non-Euclidean distance metrics besides squared distances, such as Manhattan distance.

## Elbow Method

The elbow method is used to determine the optimal number of clusters for both K-Means and K-Medoids algorithms. It plots the distortions (sum of squared distances between the data points and centroids or medoids) as a function of the number of clusters (K). The "elbow point" (where the curve changes its slope) provides an estimation of the optimal number of clusters.

In this implementation, the elbow method is demonstrated in `example_kmeans.py` and `example_kmedoids.py` and plotted as `elbow_method.png`.

## Implementations

The code snippets below provide an overview of the K-Means (`kmeans.py`) and K-Medoids (`kmedoids.py`) implementations:

- K-Means: `initialize_centroids(data, K)`, `assign_points(data, centroids)`, `get_new_centroids(data, cluster_labels, K)`, and `kmeans(data, K, max_iterations=100, tolerance=1e-4)`
- K-Medoids: `initialize_medoids(data, K)`, `assign_points(data, medoids)`, `get_new_medoids(data, cluster_labels, K)`, and `kmedoids(data, K, max_iterations=100, tolerance=1e-4)`

For visualizing the clustering results, use the `plot_clusters(data, centroids, cluster_labels, K, figsize=(12,9), filename='clusters.png')` function.

Example usage of K-Means and K-Medoids can be found in `example_kmeans.py` and `example_kmedoids.py`.
