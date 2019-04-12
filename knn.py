import pandas as pd

data = pd.read_csv('knngraph.csv')
from sklearn.neighbors import KNeighborsClassifier

x = data.values[:,0:2]
y = data.values[:,2]
print("X:",x)
print("\nY:",y)

#Basic KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x, y) 
print("\nBasic KNN: for(4,4):",model.predict([[4,4]]))


#distance Weighted KNN
weightedknn = KNeighborsClassifier(n_neighbors=3, weights='distance')
weightedknn.fit(x,y)
print("\nWeighted KNN: for(4,4):",weightedknn.predict([[4,4]]))


#average Weighted KNN
print("\nAverage Weighted KNN")
distarray,indarray = weightedknn.kneighbors([[4,6]])
print("distarray: ", distarray)
print("indarray: ", indarray)
pos=0
neg=0
poscount=0
negcount=0
for val in indarray:
	for i in range(len(val)):
		if y[i]=='positive':
			poscount+=1
			pos=pos+distarray[0][i]
		else:
			negcount+=1
			neg=neg+distarray[0][i]


pos=float(pos/poscount)
neg=float(neg/negcount)
print("pos:", pos)
print("neg:", neg)
if pos>neg:
	print ("Class is positive")
else:
	print ("class in negative")



