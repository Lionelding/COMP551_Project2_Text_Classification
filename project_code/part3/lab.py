import numpy as np
import itertools
import numpy as np
import pandas as pd
import csv

a=[0,1,2,3,4,5,6,7.0,8,9,10.5]
print np.shape(a)
b=a[::2]
print b
print sum(b)/len(b)

print int(5!=5)

with open('Good.csv', 'wb') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	wr.writerow(a)