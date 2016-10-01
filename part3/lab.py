import numpy as np
import itertools
import numpy as np
import pandas as pd
import csv

with open('lab.csv', 'rb') as a:
    reader = csv.reader(a)
    your_list = list(reader)

size=len(your_list)
print "The shape of list1 is : "+str(np.shape(your_list))
print your_list

good=[]
for i in range (0,size):
	good.extend(your_list[i])

#list1=map(list, zip(*your_list))
print "The shape of good is : "+str(np.shape(good))
print good

x = np.array([1, 2, 3, 4])
new=x.transpose
print "The shape of array is : "+str(np.shape(new))
print new


a = np.array([5,4])[np.newaxis]
print np.shape(a)
print np.shape(a.T)