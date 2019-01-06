# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 20:44:06 2018

@author: Anirudh
"""

import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
# Load Iris dataset and strip off the Labels for clustering
iris_dataset = pd.read_csv('iris_data.txt',header=None)
iris_dataset_X = np.array(iris_dataset.iloc[0:,0:4])
# Prompt the User for the choice of number of clusters
K = int(input("Enter the number of clusters K (1 - 9 only):"))
# Initialize K centroids
k_centroids = []
l = random.sample(range(0,150),K)
for i in range(0,K):
    k_centroids.append(iris_dataset_X[i])
old_centroids = k_centroids
#itr = 0
# Steps 2 and 3 of the K means algorithm, until convergence, assign K cluster labels to datapoints and re calculate cluster centers
while 1:
#      itr += 1
      cluster_label = []
      for i in range (0,len(iris_dataset_X)):
          ed = []
          for j in range(0,len(old_centroids)):
              ed.append(np.linalg.norm(iris_dataset_X[i]-old_centroids[j]))
          cluster_label.append(np.argmin(ed)) #calculate nearest cluster for each data point and assign it to that cluster
      new_cent = []
      for i in range(0,K):
          r = np.where(np.array(cluster_label) == i)[0]
          temp = [iris_dataset_X[m] for m in r ]
          new_cent.append(np.average(temp,axis=0))
      if np.array_equal(old_centroids,new_cent):
          break
      old_centroids = new_cent # until centroids doesnt change
cluster_centers = np.array(old_centroids)
print("The cluster labels (0 to K-1) for the iris dataset is: ",cluster_label)
print("The cluster centers are:",cluster_centers)

print("------------------------------The cluster plot----------------------------")
# Plot the cluster plot for K clusters
fig, ax = plt.subplots()
for i in range(K):
    points = np.array([iris_dataset_X[j] for j in range(len(iris_dataset_X)) if cluster_label[j] == i])
    ax.scatter(points[:, 0], points[:, 1])
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],marker='*', s=200, c='#050505')

    


    

