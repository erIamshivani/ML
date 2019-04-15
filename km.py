import numpy as np
import matplotlib.pyplot as plt

X=[[0.1,0.6],[0.15,0.71],[0.08,0.9],[0.16,0.85],[0.2,0.3],[0.25,0.5],[0.24,0.1],
   [0.3,0.2]]
                 
centers=np.array([[0.1,0.6],[0.3,0.2]])

print ('Initial Centroids:')
print (centers)

from sklearn.cluster import KMeans
model=KMeans(n_clusters=2,init=centers,n_init=1)
model.fit(X)
print ('\nLabels:', model.labels_)
print ('\nP6 belongs to cluster :' , model.labels_[5])
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
print (model.cluster_centers_)
plt_c1=np.asarray(c1)
plt_c2=np.asarray(c2)
plt.scatter(plt_c1[:, 0], plt_c1[:, 1])
plt.scatter(plt_c2[:, 0], plt_c2[:, 1])
plt.show()
