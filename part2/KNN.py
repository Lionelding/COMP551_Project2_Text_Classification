import pandas as pd
import numpy as np
import csv
import collections 
import re
import operator
import math

#import nltk packages to transform all vocabulary into their root expressions
from nltk.stem import PorterStemmer

#create empty list to read stop words csv file
stopword=[]

#read stopwords from csv and save it into stopword list
with open('stop.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        stopword =stopword+row

#set different values to the following parameters

#number of the most frequent words that will be used as feature words
numKeywords = 200

#k-nearest neighbors
kNearest = 55

#total number of examples in training data
numTotTraining =88639 # 647*137

#k-fold cross validation
kFold = 137

# a single list to store all words in all abstract 
abstractList =[]

# a single list that will be part of abstractList selected as traning data during the
# k-fold cross validation process
# it is used to extract feature words
allabst=[]

# flag used to skip first row in training input excel file
skip = 1 

# read training abstract
# remove stopwords
# and transform each word into its root expression
with open('train_in.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if skip == 1:
            skip = 0
        else:
            X = np.array([row])
            keyword = re.findall(r'\w+', X[0,1])
            keywordnew=[x for x in keyword if x not in stopword]
            stemmer = PorterStemmer()
            key_stemmed = [stemmer.stem(kw) for kw in keywordnew]
            abstractList.append(key_stemmed)

#reset the flag for future use
skip = 1

#create an empty list to store output (class) results
outputlist =[]

#read train_out csv file and store the info for whole training data into outputlist
with open('train_out.csv', 'rb') as csvfile:
    readerClasses = csv.reader(csvfile, delimiter=',')
    for row in readerClasses:
        if skip == 1:
            skip = 0
        else:
            X = np.array([row])

            outputlist.append(X[0,1])
# if print: outputlist is a list of of size = number of instances
 
# create several empty arrays for storing training/testing inputs/outputs for each 
# round of k-fold cross validation
emp=[]
trainingX=np.asarray(emp)
testingX =np.asarray(emp)
trainingY=[]
testingY =[]

numPerSet = numTotTraining/kFold
# initialize variable to store average error value at the end of cross validation
avgError = 0

#print len(outputlist)
# start to implement k fold  cross validation
for index in range(0,kFold):
    
    # store outputs of training data and combine abstract of all training data into allabst[]  
    for j in range(0,index*numPerSet):
        allabst=allabst+abstractList[j]
        trainingY.append(outputlist[j])
       
    # store outputs of training data
    for j in range((index+1)*numPerSet,kFold*numPerSet):
        allabst=allabst+abstractList[j]
        trainingY.append(outputlist[j])
   
    #find out the most frequent words in allabst[] and store in idx[] as feature words
    wordFreq = collections.Counter(allabst)
    mostFreq = wordFreq.most_common(numKeywords)
    #print mostFreq
    c = np.array(mostFreq)
    idx = np.unique(c[:,0].reshape((numKeywords,1)))
    
    #trainingX is of row size = idx, and col size = number of training data in this round of k-fold
    trainingX = np.zeros((idx.shape[0],numPerSet*(kFold-1)))

    for j in range(0,index*numPerSet):
        for i in range(idx.shape[0]):
        #abstractList is a list with j element
            if idx[i] in abstractList[j]:
                # if the word idx[] is present in this abstract of person j
                # the corresponding position in trainingX[i][j] will no longer to be zero
                trainingX[i][j] =  1#abstractList[j].count(idx[i])
   
    for j in range((index+1)*numPerSet,kFold*numPerSet):
        for i in range(idx.shape[0]):
        #abstractList is a list with j element
            if idx[i] in abstractList[j]:
                # if the word idx[] is present in this abstract of person j
                # the corresponding position in trainingX[i][j] will no longer to be zero
                trainingX[i][j-numPerSet] =  1#abstractList[j].count(idx[i])

    testingX = np.zeros((idx.shape[0],numPerSet))
    for j in range(index*numPerSet,(index+1)*numPerSet):
        for i in range(idx.shape[0]):
        #abstractList is a list with j element
            if idx[i] in abstractList[j]:
                # if the word idx[] is present in this abstract of person j
                # the corresponding position in testing[i][j] will no longer to be zero
                testingX[i][j-index*numPerSet] =  1#abstractList[j].count(idx[i])
        #construct testing set output data
        testingY.append(outputlist[j])

    curError =0
    # create variable for evaluation:
    TP=0;
    FP=0;
    FN=0;

    #for each sample in testing data, find out their nearest neighbors
    for i in range (0,testingX.shape[1]):
        distance =[]
        for j in range(0,trainingX.shape[1]):
            diff = testingX[:,i]- trainingX[:,j]
            sumsquare=np.sum(np.square(diff))
            distance.append((j,sumsquare))
        distance.sort(key=operator.itemgetter(1))
        neighbors =[]
        for x in range(kNearest):
            neighbors.append(distance[x][0])
        # put classes of all neigbour together to get average
        nearestClass =[] 
        for neibIdx in range(kNearest):
            tempIdx = neighbors[neibIdx]
            nearestClass.append(trainingY[tempIdx])
        # find out the majority class from the classes of all its neighbors
        countNear = collections.Counter(nearestClass)
        majorClassList = countNear.most_common(1)
        majorClassArr = np.array(majorClassList)
        majorClass=majorClassArr[0,0]
        #print 'predic Class {}'.format(majorClass)

        realClass=testingY[i]
        #print 'true Class {}'.format(realClass)
        #print majorClass==realClass
     
        #if no match: error increment by 1
        if majorClass!=realClass:
            curError += 1

        #store performance evaluation info
        #if(majorClass=='cs' and realClass=='cs'):
        #    TP+=1
        #if(majorClass=='cs' and realClass!='cs'):
        #    FP+=1
        #if(majorClass!='cs' and realClass=='cs'):
        #    FN+=1
    
    #print performance evaluation info to screen
    #print 'TP is {}'.format(TP)
    #print 'FP is {}'.format(FP)
    #print 'FN is {}'.format(FN)
    #print 'Precision is {}'.format(0.1*TP/(TP+FP)/0.1)
    #print 'Recall is {}'.format(0.1*TP/(TP+FN)/0.1)

    #give the rate of error (by dividing the total curError number by the total number of testing set)
    print 'curError is {}'.format(curError)
    curKFoldRoundErrorRate = 0.1*curError/testingX.shape[1]/0.1
    print 'curKFoldRoundErrorRate is {}'.format(curKFoldRoundErrorRate)
    avgError += curKFoldRoundErrorRate
    #clear up the arrays for next round of k-fold validation
    trainingX=np.asarray(emp)
    testingX =np.asarray(emp)
    trainingY=[]
    testingY =[]

print'average arror after k is'
avgError /=  kFold
print avgError






