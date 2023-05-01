# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 21:18:37 2023

@author: Cevan
"""
from kmeans import *
import matplotlib.pyplot as plt

data = generate_toy_data()
K_range = 10

# Apply elbow method to find the best K value
elbow_method(data, K_range)

# Set K to the optimal value from elbow method
K = 4

centroids, cluster_labels = kmeans(data, K)
plot_clusters(data, centroids, cluster_labels, K)