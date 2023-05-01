# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 22:04:02 2023

@author: Cevan
"""
import numpy as np

def initialize_medoids(data, K):
    """
    Initialize K medoids randomly from the data.
    """
    random_rows = np.random.choice(data.shape[0], K, replace=False)
    medoids = data[random_rows, :]
    return medoids

def assign_points(data, medoids):
    """
    Assign each point to the nearest medoid.
    """
    norms = np.linalg.norm(data[:, np.newaxis, :] - medoids, axis=-1)
    cluster_labels = np.argmin(norms, axis=1)
    return cluster_labels

def get_new_medoids(data, cluster_labels, K):
    """
    Compute the new medoids of the clusters.
    """
    new_medoids = []
    for k in range(K):
        cluster_data = data[cluster_labels == k]
        distance_matrix = np.sum(np.abs(cluster_data[:, np.newaxis] - cluster_data), axis=2)
        new_medoids.append(cluster_data[np.argmin(np.sum(distance_matrix, axis=1))])
    return np.array(new_medoids)

def kmedoids(data, K, max_iterations=100, tolerance=1e-4):
    """
    Perform K-medoids clustering with the specified number of clusters (K),
    stopping after the distances between the medoids drops below the tolerance.
    """
    medoids = initialize_medoids(data, K)
    for _ in range(max_iterations):
        cluster_labels = assign_points(data, medoids)
        new_medoids = get_new_medoids(data, cluster_labels, K)
        if np.linalg.norm(new_medoids - medoids) < tolerance:
            break
        medoids = new_medoids
    return medoids, cluster_labels
