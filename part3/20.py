# Import libraries
import numpy as np
import pandas as pd
import csv

import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

########################################################################################
## Load the train_in to a List
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
test_in_num=len(your_list3)
goodinput2=[]
for i in range(0,test_in_num):
	goodinput2.extend(your_list3[i])
print "The shape of Validate_in is : "+str(np.shape(goodinput2))
print 'Number of Validating_in samples is: ' +str(test_in_num)


## Load the validate_out to a List
with open('validate_out.csv', 'rb') as d:
    reader4 = csv.reader(d)
    your_list4 = list(reader4)
test_out_num=len(your_list4)
goodtarget2=[]
for i in range(0,test_out_num):
	goodtarget2.extend(your_list4[i])
print "The shape of Validate_out is : "+str(np.shape(goodtarget2))
print 'Number of Validating_out samples is: ' +str(test_out_num)

# split a training set and a test set
y_train=goodtarget
y_test=goodtarget2

categories = ['math','cs','stat','physics']

########################################################################################
print("Feature Extracting of training data")
t0 = time()
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(goodinput)
duration = time() - t0
print("n_samples: %d, n_features: %d" % X_train.shape)
print()


print("Feature Extracting of testing data")
t0 = time()
X_test = vectorizer.transform(goodinput2)
duration = time() - t0
print("n_samples: %d, n_features: %d" % X_test.shape)
print()


# mapping from integer feature name to original token string
feature_names = vectorizer.get_feature_names()
if feature_names:
    feature_names = np.asarray(feature_names)

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 100)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    print pred
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
    	print("For each category, 15 most frequent words are : ")
    	for i, category in enumerate(categories):
        	top15 = np.argsort(clf.coef_[i])[-15:]
        	print(trim("%s: %s" % (category, " ".join(feature_names[top15]))))
        	
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

###############################################################################
results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (MultinomialNB(alpha=.01), "Naive Bayes"),
        (Pipeline([('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),('classification', LinearSVC())]), "LinearSVC with L1-based feature selection"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive")):
        #(KNeighborsClassifier(n_neighbors=10), "kNN"),
        #(RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 100)
    print(name)
    results.append(benchmark(clf))



# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time", color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)
plt.show()
