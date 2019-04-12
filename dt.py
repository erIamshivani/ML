import pandas as pd

buy = pd.read_csv('buys.csv')

#from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier

x_fe=['age','income ','gender','marital_status']
x=buy[x_fe]
y=buy['buys']
print(x)
from sklearn import preprocessing
x=x.apply(preprocessing.LabelEncoder().fit_transform)
print()
print(x)

clf = DecisionTreeClassifier(criterion='entropy')
model = clf.fit(x,y)

from IPython.display import Image  
from sklearn import tree
import pydotplus
dot_data = tree.export_graphviz(model, out_file=None, 
                                feature_names=x_fe,
                                class_names=['no', 'yes'],
                                filled = True) #to export in picture format

graph = pydotplus.graph_from_dot_data(dot_data)  

Image(graph.create_png())
graph.write_png("dtree.png")

'''
criterion : string, optional (default=”gini”)
The function to measure the quality of a split. 
Supported criteria are “gini” for the Gini impurity and 
“entropy” for the information gain.


done using id3, entropy and info gain.
 ID3 algorithm uses entropy to calculate the homogeneity of a sample.
 If the sample is completely homogeneous the entropy is zero and 
 if the sample is equally divided then it has entropy of one.

E(s) = summation -pi.log(base 2)pi
gain(t,x) = etropy(T) - entropy(t,x)
A branch with entropy of 0 is a leaf node
''
