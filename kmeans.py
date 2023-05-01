# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 21:16:58 2023

@author: Cevan
"""
import numpy as np

def initialize_centroids(data, K):
    """
    Initializes centroids for the K-means algorithm by selecting K random data points.
    
    Parameters:
    data (numpy.ndarray): Input data array of shape (N, D) where N is the number of samples and D is the dimensionality.
    K (int): The number of clusters.

    Returns:
    numpy.ndarray: An array of K randomly selected initial centroids of shape (K, D).
    """
    random_rows = np.random.choice(data.shape[0], K, replace=False)
    centroids = data[random_rows,:]
    return centroids

def assign_points(data, centroids):
    """
    Assigns each data point to the nearest centroid, creating clusters.
    
    Parameters:
    data (numpy.ndarray): Input data array of shape (N, D) where N is the number of samples and D is the dimensionality.
    centroids (numpy.ndarray): Current centroids of shape (K, D) where K is the number of clusters.

    Returns:
    numpy.ndarray: An array of cluster labels for each data point of shape (N,).
    """
    norms = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=-1)
    cluster_labels = np.argmin(norms, axis=1)
    return cluster_labels

def get_new_centroids(data, cluster_labels, K):
    """
    Computes the new centroids as the mean of all data points in each cluster.

    Parameters:
    data (numpy.ndarray): Input data array of shape (N, D) where N is the number of samples and D is the dimensionality.
    cluster_labels (numpy.ndarray): An array of cluster labels for each data point of shape (N,).
    K (int): The number of clusters.

    Returns:
    numpy.ndarray: Updated centroids of shape (K, D).
    """
    new_centroids = np.array([data[cluster_labels == k].mean(axis=0) for k in range(K)])
    return new_centroids

def kmeans(data, K, max_iterations=100, tolerance=1e-4):
    """
    Performs the K-means clustering algorithm on the input data.

    Parameters:
    data (numpy.ndarray): Input data array of shape (N, D) where N is the number of samples and D is the dimensionality.
    K (int): The number of clusters.
    max_iterations (int, optional): Maximum number of iterations for the K-means algorithm. Default is 100.
    tolerance (float, optional): Convergence tolerance for the K-means algorithm. Default is 1e-4.

    Returns:
    tuple: A tuple containing the final centroids (numpy.ndarray) of shape (K, D) and the cluster labels (numpy.ndarray) of shape (N,).
    """
    centroids = initialize_centroids(data, K)
    for _ in range(max_iterations):
        cluster_labels = assign_points(data, centroids)
        new_centroids = get_new_centroids(data, cluster_labels, K)
        if np.linalg.norm(new_centroids - centroids) < tolerance:
            break
        centroids = new_centroids
    return centroids, cluster_labels

def generate_toy_data(num_clusters=4, samples_per_cluster=100, std_dev=1):
    """
    Generates a toy dataset with specified number of clusters, samples per cluster, and standard deviation.

    Parameters:
    num_clusters (int, optional): Number of clusters in the dataset. Default is 4.
    samples_per_cluster (int, optional): Number of samples per cluster. Default is 100.
    std_dev (float, optional): Standard deviation for the Gaussian distribution of each cluster. Default is 1.

    Returns:
    numpy.ndarray: A generated dataset of shape (num_clusters * samples_per_cluster, 2).
    """
    centroids = np.random.normal(0, 4, size=(num_clusters, 2))
    dataset = np.vstack([np.random.normal(loc=centroid, scale=std_dev, size=(samples_per_cluster, 2)) for centroid in centroids])
    return dataset

def plot_clusters(data, centroids, cluster_labels, K, figsize=(12,9), filename='clusters.png'):
    """
    Plots the clustered data points and centroids using matplotlib.

    Parameters:
    data (numpy.ndarray): Input data array of shape (N, D) where N is the number of samples and D is the dimensionality.
    centroids (numpy.ndarray): Centroids of shape (K, D) where K is the number of clusters.
    cluster_labels (numpy.ndarray): An array of cluster labels for each data point of shape (N,).
    K (int): The number of clusters.
    figsize (tuple, optional): Figure size for the matplotlib plot. Default is (12, 9).
    filename (str, optional): Filename for saving the generated plot. Default is 'clusters.png'.

    Returns:
    None
    """
    import matplotlib.pyplot as plt 

    colors = plt.cm.viridis(np.linspace(0, 1, K))
    fig, ax = plt.subplots(figsize=figsize)

    for k in range(K):
        cluster_data = data[cluster_labels == k]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], color=colors[k], s=25)
        ax.scatter(centroids[k, 0], centroids[k, 1], color=colors[k], marker='*', s=500)

    ax.set_title(f"K={K}")
    plt.savefig(filename)
    plt.close(fig)

def elbow_method(data, K_range):
    """
    Determines the optimal number of clusters for the K-means algorithm using the elbow method. Plots the distortion as a function of K.

    Parameters:
    data (numpy.ndarray): Input data array of shape (N, D) where N is the number of samples and D is the dimensionality.
    K_range (int): The maximum number of clusters to consider for the elbow method.

    Returns:
    None
    """
    import matplotlib.pyplot as plt 
    distortions = []
    K_values = range(1, K_range+1)

    for K in K_values:
        centroids, cluster_labels = kmeans(data, K)
        distortions.append(np.sum(np.min(np.sqrt(np.sum((data - centroids[cluster_labels])**2, axis=1)), axis=0)))

    plt.figure()
    plt.plot(K_values, distortions, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method')
    plt.savefig('elbow_method.png')
    plt.close()
