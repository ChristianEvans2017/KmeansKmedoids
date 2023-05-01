# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 22:05:06 2023

@author: Cevan
"""
from kmeans import generate_toy_data, elbow_method, plot_clusters
from kmedoids import *
import matplotlib.pyplot as plt

data = generate_toy_data()
K_range = 10

# Apply elbow method to find the best K value
elbow_method(data, K_range)

# Set K to the optimal value from elbow method
K = 4

medoids, cluster_labels = kmedoids(data, K)
plot_clusters(data, medoids, cluster_labels, K, filename='kmedoids_clusters.png')
