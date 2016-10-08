import numpy as np
import nltk
import pickle
import csv
from collections import defaultdict, Counter
import itertools
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
import math
import operator

def main():
    training_file_path_d = "/Users/Caitrin/Desktop/COMP551_Project2_Text_Classification/dataset/train_in.csv"
    training_file_path_c = "/Users/Caitrin/Desktop/COMP551_Project2_Text_Classification/dataset/train_out.csv"
    class_documents = get_training_class_documents_from_file(training_file_path_d, training_file_path_c)
    print "got class documents"
    del class_documents['category']
    class_docs, class_terms, vocabulary = get_class_term_counts(class_documents)
    print "got class term counts"
    prior, condProb = train_multinomial_naive_bayes(class_terms, class_docs, vocabulary)
    print "calculated prior and condprob"
    testDocuments = get_test_documents("/Users/Caitrin/Desktop/COMP551_Project2_Text_Classification/dataset/test_in.csv")

    for document in testDocuments:
        c = get_document_max_class(prior, condProb, vocabulary, document)
        print document, c
        
def get_test_documents(test_document_path):
    """ Returns dict of test documents
    """
    documents = dict()
    with open(test_document_path, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            documents[line[0]] = line[1]

    return documents
    
def get_document_max_class(prior, condProb, vocabulary, document):
    """
    Use previously generated priors and probabilities to find most probabable class
    given a document
    prior prior
    condProb condProb
    class_terms classterms
    vocabulary vocabulary
    document string of document
    """
    porter_stemmer = PorterStemmer()
    terms = [porter_stemmer.stem(x.lower()) for x in word_tokenize(document)]
    score = dict()
    for c in prior:
        score[c] = math.log(prior[c])
        for term in terms:
            if term in vocabulary:
                score[c] += condProb[term][c]
        print score, c
    return max(score.iteritems(), key=operator.itemgetter(1))[0]
    
def train_multinomial_naive_bayes(class_terms, class_docs, vocabulary):
    """Train naive bayes
    Returns:
       prior for each class - dict()
       condProb p(term|class) - nested dictionaries
    """
    prior = dict()
    condProb = defaultdict(lambda: dict())
    number_of_docs = sum(class_docs.itervalues())
    
    for c in class_terms:
        
        prior[c] = float(class_docs[c])/float(number_of_docs)
        
        # len is like adding 1 to every value
        all_term_count = sum(class_terms[c].itervalues()) + len(class_terms[c])
        
        for t in vocabulary:
            term_count = 1 
            if t in class_terms[c]:
                term_count += class_terms[c][t]
            condProb[t][c] = float(term_count)/float(all_term_count)

    return prior, condProb

def get_class_term_counts(class_documents):
    """ Count term by class
    Returns:
       class_docs number of docs per class - dict()
       class_terms nested dictionary of terms counted by class - dict()
       W set of all possible terms - set()
    """
    class_docs = dict()
    class_terms = dict()
    w = set()
    porter_stemmer = PorterStemmer()
    for c in class_documents:
        class_docs[c] = len(class_documents[c])
        terms = list()
        for d in class_documents[c]:
            terms.extend(word_tokenize(d))
        class_terms[c] = Counter([porter_stemmer.stem(x.lower()) for x in terms])
        w.update(class_terms[c].iterkeys())

    return class_docs, class_terms, w


def get_training_class_documents_from_file(training_file_path_d, training_file_path_c):
    """Gets docs from csv into python objects
    """
    documents = dict()
    with open(training_file_path_d, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            documents[line[0]] = line[1]

    class_documents = defaultdict(lambda: list())
    with open(training_file_path_c, 'rb') as csvfile1:
        reader2 = csv.reader(csvfile1)
        for line in reader2:
            class_documents[line[1]].append(documents[line[0]])

    return class_documents

if __name__ == "__main__":
    main()
