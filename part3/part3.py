## Import libraries
import numpy as np
import pandas as pd
import csv
import sys
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics
global count
global method


########################################################################################
## Initializing variables
k_fold=5
method_num=2
categories = ['math','cs','stat','physics']

########################################################################################
## Load the train_in to a List
print "_"*100
print "------Loading Data-----"
print ""
with open('train_in.csv', 'rb') as a:
    reader = csv.reader(a)
    your_list = list(reader)
train_in_num=len(your_list)
goodinput=[]
for i in range(0,train_in_num):
	goodinput.extend(your_list[i])
print "The shape of Train_in is:" + str(np.shape(goodinput))
print 'Number of Training_in samples is: ' +str(train_in_num)


## Load the train_out to a List
with open('train_out.csv', 'rb') as b:
    reader2 = csv.reader(b)
    your_list2 = list(reader2)
train_out_num=len(your_list2)
goodtarget=[]
for i in range(0,train_out_num):
	goodtarget.extend(your_list2[i])
print "The shape of Train_out is : "+str(np.shape(goodtarget))
print 'Number of Training_out results is: ' +str(train_out_num)


## Load the validate_in to a List
with open('validate_in.csv', 'rb') as c:
    reader3 = csv.reader(c)
    your_list3 = list(reader3)
validate_in_num=len(your_list3)
goodinput2=[]
for i in range(0,validate_in_num):
	goodinput2.extend(your_list3[i])
print "The shape of Validate_in is : "+str(np.shape(goodinput2))
print 'Number of Validating_in samples is: ' +str(validate_in_num)


## Load the validate_out to a List
with open('validate_out.csv', 'rb') as d:
    reader4 = csv.reader(d)
    your_list4 = list(reader4)
validate_out_num=len(your_list4)
goodtarget2=[]
for i in range(0,validate_out_num):
	goodtarget2.extend(your_list4[i])
print "The shape of Validate_out is : "+str(np.shape(goodtarget2))
print 'Number of Validating_out samples is: ' +str(validate_out_num)


## Load the test_in to a List for the real predication 
with open('test_in.csv', 'rb') as e:
    reader5 = csv.reader(e)
    your_list5 = list(reader5)
test_in_num=len(your_list5)
goodinput3=[]
for i in range(0,test_in_num):
	goodinput3.extend(your_list5[i])
print "The shape of test_in is : "+str(np.shape(goodinput3))
print 'Number of test_in samples is: ' +str(test_in_num)

print ""
print "-----Data Loaded-----"
print ""

###############################################################################
## Defining training and testing method 
def train_and_test(method_k):
    print('')
    print("Training: ")
    method_k.fit(X_train, y_train)
    pred = method_k.predict(X_train)
    grade = metrics.accuracy_score(y_train, pred)
    print("accuracy:   %0.3f" % grade)
    accuracy.append(grade)
    return grade

method=0

## Defining the pridiction method
def predict(method_k):
    print('')
    print("Training: ")
    method_k.fit(X_train, y_train)
    pred = method_k.predict(X_test)
    print 'size of pred is : '+ str(np.shape(pred))
    return pred


########################################################################################
## k_fold testing 
samples = [0]
sample_len=train_in_num/k_fold
for i in range (1,k_fold+1):
	samples.append(sample_len*i)

print "samples is : "+str(samples)
print "k_fold number is : " + str(k_fold)

accuracy=[]
method_accuracy=[]
count=0

for i in range (0, k_fold):
	print('_' * 100)
	print "-----Step "+ str(i)+ " :Starting k_fold testing-----"
	print ""
	test_k_in=goodinput[samples[i]:samples[i+1]]
	test_k_out=goodtarget[samples[i]:samples[i+1]]
	train_k_in=int(i!=0)*goodinput[0:sample_len*i] + int(i!=(k_fold-1))*goodinput[samples[i+1]:samples[i+1]+sample_len*(k_fold-i-1)]
	train_k_out=int(i!=0)*goodtarget[0:sample_len*i] + int(i!=(k_fold-1))*goodtarget[samples[i+1]:samples[i+1]+sample_len*(k_fold-i-1)]
	print "train_k_in start from : " + str(0) +" to :" + str(sample_len*i)
	print "and train_k_in start from : " + str(int(i!=(k_fold-1))*samples[i+1]) + " to :" + str(int(i!=(k_fold-1))*(samples[i+1]+sample_len*(k_fold-i-1)-1))
 	print "train_k_in shape is : " + str(np.shape(train_k_in))
 	print "test_k_in shape is : " + str(np.shape(test_k_in))

# Load test and train targets
	y_train=train_k_out
	y_test=test_k_out

# ########################################################################################
## Extracting Feature from training data
	print("Extracting Feature from training data")
	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
	X_train = vectorizer.fit_transform(train_k_in)
	print("Samples: %d, Features: %d" % X_train.shape)
	print ""

## Extracting Feature from testing data
	print("Extracting Feature from testing data")
	X_test = vectorizer.transform(test_k_in)
	print("Samples: %d, Features: %d" % X_test.shape)
	print ""

# ###############################################################################

## Call the training and testing method to see the results
	results = []
	for method_k, method_name in (
	        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
	        (PassiveAggressiveClassifier(n_iter=1), "Passive-Aggressive")):
	    print('')
	    print(method_name)
	    method_accuracy.append(method_name)
	    results.append(train_and_test(method_k))

# ###############################################################################

## Summarize the traing and testing results
print('_' * 100)
print "-----Step " + str(k_fold) + " :Summarize the accuracy using each method-----"
print ""
print accuracy
k_fold_testing=[]
for i in range (0,method_num):
	a=method_accuracy[i]
	b=sum(accuracy[i::method_num])/k_fold
	k_fold_testing.append(a)
	k_fold_testing.append(b)
print ''
print 'Short Summary of results:'
print k_fold_testing

###############################################################################

## Start predicting
print('_' * 100)
print "-----Step " + str(k_fold+1) + " :Prediction -----"
print ""
y_train=goodtarget
print("Extracting Features from full training data")
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(goodinput)
print("Samples: %d, Features: %d" % X_train.shape)
print ""


print("Extracting Features from full testing data")
X_test = vectorizer.transform(goodinput3)
print("Samples: %d, Features: %d" % X_test.shape)
print ""

good=predict(RidgeClassifier(tol=1e-2, solver="lsqr"))
print 'size of good is : '+ str(np.shape(good))
print 'Type : '+ str(type(good))
print good

# Print the output result to a csv file
resultFile = open("output.csv",'wb')
wr = csv.writer(resultFile,dialect='excel')
wr.writerows(good)
