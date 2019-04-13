import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X=[[0.1,0.6],[0.15,0.71],[0.08,0.9],[0.16,0.85],[0.2,0.3],[0.25,0.5],[0.24,0.1],
   [0.3,0.2]]
                 
centers=np.array([[0.1,0.6],[0.15,0.71]])

print ('Initial Centroids:')
print (centers)

from sklearn.cluster import KMeans
model=KMeans(n_clusters=2,init=centers,n_init=1)
model.fit(X)
print ('\nLabels:', model.labels_)
print ('\nP6 belongs to cluster :' , model.labels_[5])
print ('\nPopulation around cluster 2 :' ,np.count_nonzero(model.labels_==0))
print ('Population around cluster 2 :' ,np.count_nonzero(model.labels_==1))
print ('\nNew Centroids are :')
print (model.cluster_centers_)

'''
OUTPUT 1st 2nd:

Initial Centroids:
[[0.1  0.6 ]
 [0.15 0.71]]

Labels: [1 1 1 1 0 0 0 0]

P6 belongs to cluster : 0

Population around cluster 2 : 4
Population around cluster 2 : 4

New Centroids are :
[[0.2475 0.275 ]
 [0.1225 0.765 ]]

 
OUTPUT 1st last:

Initial Centroids:
[[0.1 0.6]
 [0.3 0.2]]

Labels: [0 0 0 0 1 0 1 1]

P6 belongs to cluster : 0

Population around cluster 2 : 5
Population around cluster 2 : 3

New Centroids are :
[[0.148      0.712     ]
 [0.24666667 0.2       ]]
 
 '''