import numpy as np
import nltk
import pickle
import csv
from collections import defaultdict, Counter

def main():
    training_file_path_d = "/Users/Caitrin/Desktop/COMP551_Project2_Text_Classification/dataset/train_in.csv"
    training_file_path_c = "/Users/Caitrin/Desktop/COMP551_Project2_Text_Classification/dataset/train_out.csv"
    class_documents = get_training_class_documents_from_file(training_file_path_d, training_file_path_c)
    del class_documents['category']
    # class_terms, vocabulary = get_class_term_counts(class_documents)
    # prior, condProb = train_multinomial_naive_bayes(class_terms, vocabulary)
    

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
    pass

def get_training_class_documents_from_file(training_file_path_d, training_file_path_c):
    documents = dict()
    with open(training_file_path_d, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            documents[line[0]] = [line[1]]

    class_documents = defaultdict(lambda: list())
    with open(training_file_path_c, 'rb') as csvfile1:
        reader2 = csv.reader(csvfile1)
        for line in reader2:
            class_documents[line[1]].append(documents[line[0]])

    return class_documents

if __name__ == "__main__":
    main()
