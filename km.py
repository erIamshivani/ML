# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 20:34:29 2019

@author: Ashwini Bagad
"""

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
         
data = pd.read_csv('kmns.csv')
X= np.asarray(data)
print("X:",X)
print("Enter indexes between 0 and 7 for cent: ")
a= int(input())
b= int(input())
centers=np.array([X[a],X[b]])

print ('Initial Centroids:')
print (centers)

from sklearn.cluster import KMeans
model=KMeans(n_clusters=2,init=centers,n_init=1)
model.fit(X)
print ('\nLabels:', model.labels_)
print ('\nP6 belongs to cluster :' , model.labels_[4])
print ('\nPopulation around cluster 1 :' ,np.count_nonzero(model.labels_==0))
print ('Population around cluster 2 :' ,np.count_nonzero(model.labels_==1))
c1=[]
c2=[]
for i in range(len(model.labels_)):
    if (model.labels_[i]==0):
        c1.append(X[i]) 
    else:
        c2.append(X[i])
        
print("Cluster 1:", c1)
print("Cluster 2:", c2)
print ('\nNew Centroids are :')
newcent= model.cluster_centers_
print (newcent)
plt_c1=np.asarray(c1)
plt_c2=np.asarray(c2)
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(plt_c1[:, 0], plt_c1[:, 1])
plt.scatter(plt_c2[:, 0], plt_c2[:, 1])
plt.scatter(newcent[0], newcent[1], c='r',s=30)
plt.show()