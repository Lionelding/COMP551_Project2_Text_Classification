# Import libraries
import numpy as np
import pandas as pd
import csv
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
global count
global method


########################################################################################
#Initializing variables
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


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

###############################################################################

def benchmark(clf):
    print('_' * 100)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_train)
    print pred
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_train, pred)
    print("accuracy:   %0.3f" % score)
    accuracy.append(score)

    if hasattr(clf, 'coef_'):
    	print("For each category, 15 most frequent words are : ")
    	for i, category in enumerate(categories):
        	top15 = np.argsort(clf.coef_[i])[-15:]
        	print(trim("%s: %s" % (category, " ".join(feature_names[top15]))))
        	
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

method=0

def predict(clf):
    print('_' * 100)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    print 'size of pred is : '+ str(np.shape(pred))

    predict_time = time() - t0
    print("test time:  %0.3fs" % predict_time)

    # score = metrics.accuracy_score(y_train, pred)
    # print("accuracy:   %0.3f" % score)
    # accuracy.append(score)

    if hasattr(clf, 'coef_'):
    	print("For each category, 15 most frequent words are : ")
    	for i, category in enumerate(categories):
        	top15 = np.argsort(clf.coef_[i])[-15:]
        	print(trim("%s: %s" % (category, " ".join(feature_names[top15]))))
        	
    clf_descr = str(clf).split('(')[0]
    return pred


########################################################################################
#k_fold testing 
# samples = [0]
# sample_len=train_in_num/k_fold
# for i in range (1,k_fold+1):
# 	samples.append(sample_len*i)

# print "samples is : "+str(samples)
# print "k_fold number is : " + str(k_fold)
# print "number"

# accuracy=[]
# method_accuracy=[]
# count=0

# for i in range (0, k_fold):

# 	print('_' * 100)
# 	print "-----Step "+ str(i)+ " :Starting k_fold testing-----"
# 	print ""
# 	test_k_in=goodinput[samples[i]:samples[i+1]]
# 	test_k_out=goodtarget[samples[i]:samples[i+1]]
# 	train_k_in=int(i!=0)*goodinput[0:sample_len*i] + int(i!=(k_fold-1))*goodinput[samples[i+1]:samples[i+1]+sample_len*(k_fold-i-1)]
# 	train_k_out=int(i!=0)*goodtarget[0:sample_len*i] + int(i!=(k_fold-1))*goodtarget[samples[i+1]:samples[i+1]+sample_len*(k_fold-i-1)]
# 	print "train_k_in start from : " + str(0) +" to :" + str(sample_len*i)
# 	print "and train_k_in start from : " + str(int(i!=(k_fold-1))*samples[i+1]) + " to :" + str(int(i!=(k_fold-1))*(samples[i+1]+sample_len*(k_fold-i-1)-1))
#  	print "train_k_in shape is : " + str(np.shape(train_k_in))
#  	print "test_k_in shape is : " + str(np.shape(test_k_in))

# # Load test and train targets
# 	y_train=train_k_out
# 	y_test=test_k_out


# ########################################################################################
# 	print("Feature Extracting of training data")
# 	t0 = time()
# 	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
# 	X_train = vectorizer.fit_transform(train_k_in)
# 	duration = time() - t0
# 	print("n_samples: %d, n_features: %d" % X_train.shape)
# 	print ""


# 	print("Feature Extracting of testing data")
# 	t0 = time()
# 	X_test = vectorizer.transform(test_k_in)
# 	duration = time() - t0
# 	print("n_samples: %d, n_features: %d" % X_test.shape)
# 	print ""


# # mapping from integer feature name to original token string
# 	feature_names = vectorizer.get_feature_names()
# 	if feature_names:
# 	    feature_names = np.asarray(feature_names)


# ###############################################################################
# 	results = []
# 	for clf, name in (
# 	        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
# 	        #(Perceptron(n_iter=50), "Perceptron"),
# 	        #(MultinomialNB(alpha=.01), "Naive Bayes"),
# 	        #(Pipeline([('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),('classification', LinearSVC())]), "LinearSVC with L1-based feature selection"),
# 	        (PassiveAggressiveClassifier(n_iter=1), "Passive-Aggressive")):
# 	        #(KNeighborsClassifier(n_neighbors=10), "kNN")):
# 	        #(RandomForestClassifier(n_estimators=250), "Random forest")):
# 	    print('=' * 100)
# 	    print(name)
# 	    method_accuracy.append(name)
# 	    results.append(benchmark(clf))



# 	make some plots

# 	indices = np.arange(len(results))

# 	results = [[x[i] for x in results] for i in range(4)]

# 	clf_names, score, training_time, test_time = results
# 	training_time = np.array(training_time) / np.max(training_time)
# 	test_time = np.array(test_time) / np.max(test_time)

# 	plt.figure(figsize=(12, 8))
# 	plt.title("Score with k_fold : %d" % count)
# 	plt.barh(indices, score, .2, label="score", color='navy')
# 	plt.barh(indices + .3, training_time, .2, label="training time", color='c')
# 	plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
# 	plt.yticks(())
# 	plt.legend(loc='best')
# 	plt.subplots_adjust(left=.25)
# 	plt.subplots_adjust(top=.95)
# 	plt.subplots_adjust(bottom=.05)

# 	for i, c in zip(indices, clf_names):
# 	    plt.text(-.3, i, c)
	

# 	plt.savefig('k_fold_%d.png' % count)
# 	#plt.show()
# 	#plt.close()
# 	count=count+1



# ###############################################################################
# print('_' * 100)
# print "-----Step " + str(k_fold) + " :Summarize the accuracy using each method-----"
# print ""
# print accuracy
# k_fold_testing=[]
# for i in range (0,method_num):
# 	a=method_accuracy[i]
# 	b=sum(accuracy[i::method_num])/k_fold
# 	k_fold_testing.append(a)
# 	k_fold_testing.append(b)

# print k_fold_testing


###############################################################################
print('_' * 100)
print "-----Step " + str(k_fold+1) + " :Prediction -----"
print ""

y_train=goodtarget

print("Feature Extracting of full training data")
t0 = time()
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(goodinput)
duration = time() - t0
print("n_samples: %d, n_features: %d" % X_train.shape)
print ""


print("Feature Extracting of full testing data")
t0 = time()
X_test = vectorizer.transform(goodinput3)
duration = time() - t0
print("n_samples: %d, n_features: %d" % X_test.shape)
print ""


# mapping from integer feature name to original token string
feature_names = vectorizer.get_feature_names()
if feature_names:
    feature_names = np.asarray(feature_names)



results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        #(Perceptron(n_iter=50), "Perceptron"),
        #(MultinomialNB(alpha=.01), "Naive Bayes"),
        #(Pipeline([('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),('classification', LinearSVC())]), "LinearSVC with L1-based feature selection"),
        (PassiveAggressiveClassifier(n_iter=1), "Passive-Aggressive")):
        #(KNeighborsClassifier(n_neighbors=10), "kNN")):
        #(RandomForestClassifier(n_estimators=250), "Random forest")):
    	print"" 

good=predict(RidgeClassifier(tol=1e-2, solver="lsqr"))
print 'size of good is : '+ str(np.shape(good))
print 'Type : '+ str(type(good))
print good

# np.savetxt('lol.csv', good, delimiter=',',fmt='%.5f')

# lol=[]
# for i in range(0,len(good)):
# 	#print i
# 	lol.append(good[i])


resultFile = open("output2.csv",'wb')
wr = csv.writer(resultFile,dialect='excel')
wr.writerows(good)
