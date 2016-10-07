import pandas as pd
import numpy as np
import csv
import collections 
import re
import operator


import math
from nltk.stem import PorterStemmer




stopword=[]

with open('stop.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
       # print row
        stopword =stopword+row

numKeywords = 100
kNearest = 40 #40 not good
numTotTraining =1000#88639 # 647*137
kFold = 10

list=[]
# list to store all abstract as list
abstractList =[]
allabst=[]
#accumWords = np.asarray(list)

skip = 1 # to skip first row in training input excel file


with open('train_in.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if skip == 1:
            skip = 0
        else:
            #print row
            X = np.array([row])
            #print X.shape
            keyword = re.findall(r'\w+', X[0,1])
            keywordnew=[x for x in keyword if x not in stopword]
            stemmer = PorterStemmer()
            key_stemmed = [stemmer.stem(kw) for kw in keywordnew]
            #abstractList[0] is same as keyword
            abstractList.append(key_stemmed)


skip = 1
outputlist =[]
with open('train_out.csv', 'rb') as csvfile:
    readerClasses = csv.reader(csvfile, delimiter=',')
    for row in readerClasses:
        if skip == 1:
            skip = 0
        else:
            #print row
            #X is an array shape 1,2
            X = np.array([row])
            #print X[0,1]
            outputlist.append(X[0,1])
    #print outputlist # is list of size number of instances


#Create several empty arrays for storing training/testing inputs/outputs with k-fold cross validation
emp=[]
trainingX=np.asarray(emp)
testingX =np.asarray(emp)
trainingY=[]
testingY =[]

numPerSet = numTotTraining/kFold
#implement k fold 
# Initialize variable to store error value
avgError = 0

print len(outputlist)
for index in range(0,kFold):
    print 
    for j in range(0,index*numPerSet):
        allabst=allabst+abstractList[j]
        trainingY.append(outputlist[j])
       
    
    for j in range((index+1)*numPerSet,kFold*numPerSet):
        allabst=allabst+abstractList[j]
        trainingY.append(outputlist[j])
    

    #allabst_2 = [allabst_2 + stemmer.stem(kw) for kw in allabst]
    #print 'allabst'
    #print allabst
    #print 'allabst_2'
    #print allabst_2
   
    #allabst_2 is list with root ['roor','rrr']
    wordFreq = collections.Counter(allabst)
    mostFreq = wordFreq.most_common(numKeywords)
    print mostFreq
    c = np.array(mostFreq)
    idx = np.unique(c[:,0].reshape((numKeywords,1)))
 
    trainingX = np.zeros((idx.shape[0],numPerSet*(kFold-1)))
    for j in range(0,index*numPerSet):
        for i in range(idx.shape[0]):
        #abstractList is a list with j element
            if idx[i] in abstractList[j]:
                trainingX[i][j] =  abstractList[j].count(idx[i])#30
            #elif idx[i] in abstractList[j] and abstractList[j].count(idx[i])<=3 and abstractList[j].count(idx[i])>1:
            #    trainingX[i][j] = 30
            #elif idx[i] in abstractList[j] and abstractList[j].count(idx[i])<=3 and abstractList[j].count(idx[i])>1:
            #    trainingX[i][j] = 30
   
    for j in range((index+1)*numPerSet,kFold*numPerSet):
        for i in range(idx.shape[0]):
        #abstractList is a list with j element
            if idx[i] in abstractList[j]:
                trainingX[i][j-numPerSet] =  abstractList[j].count(idx[i])#30
            #elif idx[i] in abstractList[j] and abstractList[j].count(idx[i])<=3 and abstractList[j].count(idx[i])>1:
            #    trainingX[i][j-numPerSet] = 30
            #elif idx[i] in abstractList[j] and abstractList[j].count(idx[i])==1:
            #    trainingX[i][j-numPerSet] = 30
    #finish construct trainingX trainingY for this k fold round
    #Based on cross validation number, store necessary training/testing inputs/outputs info for each round of cross validation
    testingX = np.zeros((idx.shape[0],numPerSet))
    for j in range(index*numPerSet,(index+1)*numPerSet):
        for i in range(idx.shape[0]):
        #abstractList is a list with j element
            if idx[i] in abstractList[j]:
                testingX[i][j-index*numPerSet] =  abstractList[j].count(idx[i])
            #elif idx[i] in abstractList[j] and abstractList[j].count(idx[i])<=3 and abstractList[j].count(idx[i])>1:
            #    testingX[i][j-index*numPerSet] = 30
            #elif idx[i] in abstractList[j] and abstractList[j].count(idx[i])==1:
            #    testingX[i][j-index*numPerSet] = 30


        testingY.append(outputlist[j])

    curError =0
    for i in range (0,testingX.shape[1]):#range(0,testingX.shape[1]):
        distance =[]
        for j in range(0,trainingX.shape[1]):
            diff = testingX[:,i]- trainingX[:,j]
            sumsquare=np.sum(np.square(diff))
            distance.append((j,sumsquare))
        distance.sort(key=operator.itemgetter(1))
        neighbors =[]
        for x in range(kNearest):
            #print "distance"
            #print distance
            neighbors.append(distance[x][0])
        #print 'neigh for people# {}'.format(i)
        #print neighbors # the idex are number of j; col index in current training set nearest the testing instance i
    
        # grap trainng output classes to see average
        nearestClass =[] # put classes of all neigbour together to get average
        for neibIdx in range(kNearest):
            tempIdx = neighbors[neibIdx]
            nearestClass.append(trainingY[tempIdx])
        #get for current i for testingX the neigbours outputs list
        #print 'for testing idx {} neigb outpus list is'.format(i)
        #print nearestClass
        countNear = collections.Counter(nearestClass)
        majorClassList = countNear.most_common(1)
        majorClassArr = np.array(majorClassList)
        majorClass=majorClassArr[0,0]
        #print 'predic Class {}'.format(majorClass)
        #findout real testingX i class output
        realClass=testingY[i]
        #print 'true Class {}'.format(realClass)
        #print majorClass==realClass
        #to save the error for this k fold round
        

        if majorClass!=realClass:
            curError += 1
    #give the rate of error (by dividing total curError number by number of testing set)
    #print 'testingX.shape[1] is {}'.format(testingX.shape[1])
    print 'curError is {}'.format(curError)
    curKFoldRoundErrorRate = 0.1*curError/testingX.shape[1]/0.1
    print 'curKFoldRoundErrorRate is {}'.format(curKFoldRoundErrorRate)
    avgError += curKFoldRoundErrorRate

    trainingX=np.asarray(emp)
    testingX =np.asarray(emp)
    trainingY=[]
    testingY =[]

print'average arror after k is'
avgError /=  kFold
print avgError

with open('mydata.csv', 'w') as mycsvfile:
    thedatawriter = csv.writer(mycsvfile, delimiter=' ')
    thedatawriter.writerow(avgError)

 #   for x in range(len(trainingSet)):
	#	dist = euclideanDistance(testInstance, trainingSet[x], length)
	#	distances.append((trainingSet[x], dist))
	#distances.sort(key=operator.itemgetter(1))



        #def sumSquareDiff(self,X,Y, weights):
        #theoryY = np.dot(X, weights)
        #print('The predicted y is')
        #print(theoryY)
        #diff = (Y-theoryY)
        #sumSquareDiffNum = np.sum(np.square(diff))
        #return sumSquareDiffNum
    
    #for x in np.nditer(idx):
    #    print x


        #see = np.array([['xxx']['xxx']['xxx']['yyy']['yyy']['g']])
        #print see
        #print see.shape
        #vvv = np.vstack({tuple(row) for row in see})
        #print vvv




