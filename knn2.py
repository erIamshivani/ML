import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def locally_weighted(x, y):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x, y) 
    print("Enter X value : ")
    x_i = int(input())
    print("Enter Y value : ")
    y_i = int(input())    
    predict = [x_i, y_i]
    if neigh.predict([predict]) == 0:
        print("The sample "+str(predict)+" is of negative class")
    else:
        print("The sample "+str(predict)+" is of positive class")


def distance_weighted(x, y):
    neigh = KNeighborsClassifier(n_neighbors=3, weights = 'distance')
    neigh.fit(x, y)
    print("Enter new sample to predict : ")
    print("Enter X value : ")
    x_i = int(input())
    print("Enter Y value : ")
    y_i = int(input())
    predict = [x_i, y_i]
    if neigh.predict([predict]) == 0:
        print("The sample "+str(predict)+" is of negative class")
    else:
        print("The sample "+str(predict)+" is of positive class")
        
data = pd.read_csv('knngraph.csv')

x = data.values[:,0:2]
y = data.values[:,2]
print(x)
print(y)
print("----MENU-----")
print("1 - Locally weighted ")
print("2 - Distance weighted ")
print("Enter your choice : ")
user_input = int(input())
if user_input == 1:
    locally_weighted(x, y)
else:
    distance_weighted(x, y)