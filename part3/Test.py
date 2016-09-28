# Import libraries
import numpy as np
import pandas as pd
import csv

## Load the train_in to a List and then convert to an array
with open('train_in.csv', 'rb') as a:
    reader = csv.reader(a)
    your_list = list(reader)

print "The training list is "
train_in_num=len(your_list)
train_in=np.asarray(your_list)
print train_in_num

## Load the train_out to a List and then convert to an arrary
with open('train_out.csv', 'rb') as b:
    reader2 = csv.reader(b)
    your_list2 = list(reader2)
train_out_num=len(your_list2)
train_out=np.asarray(your_list2)

print train_out_num
print type(train_in[1])


from sklearn.datasets import load_iris
from sklearn import tree

clf = tree.DecisionTreeClassifier(max_features=str)
iris = train_in

clf = clf.fit(train_in, train_out)
tree.export_graphviz(clf, out_file='tree.dot') 