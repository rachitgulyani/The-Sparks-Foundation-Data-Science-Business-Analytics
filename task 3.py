# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:43:10 2021

@author: RACHIT
"""
#Importing pandas library to load the dataset
import pandas as pd

#Loading the dataset
dataset=pd.read_csv("Iris.csv")
dataset=dataset.drop('Id',axis=1)
X=dataset.iloc[:,0:4].values
y=dataset.iloc[:,4:5].values

#Training the Decision Tree Classifier on the dataset
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(X,y)

# Import necessary libraries for graph viz
from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

# Visualize the graph
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data, feature_names=dataset.columns[0:4],  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


prediction=classifier.predict([[3.5,3.1,2.6,0.2]])
print("Predicted Result="+prediction)