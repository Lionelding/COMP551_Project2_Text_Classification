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
from nltk.corpus import stopwords
import sys

def main():

    try:
        training_file_path_d = sys.argv[1]
        training_file_path_c = sys.argv[2]
    except:
        print "please specify training set path"
        exit(0)
        
    if "crossvalidate" in sys.argv[3]:
        alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0, 1]
        a, error = run_cross_validation(training_file_path_d, training_file_path_c, alpha)
        print "best: ", a, error
    elif "test" in sys.argv[3]:
        try:
            alpha = [float(sys.argv[4])]
        except:
            print "please specify alpha - the smoothing parameter"
        a, error = run_cross_validation(training_file_path_d, training_file_path_c, alpha)
    else:
        try:
            test_file_path = sys.argv[3]
            alpha = float(sys.argv[4])
        except:
            "please specify test set path and smoothing parameter if not performing cross validation"
            exit(0)
        try:
            documents, document_class = get_training_class_documents_from_file(training_file_path_d, training_file_path_c)
            testDocuments = get_test_documents(test_file_path)
        except:
            print "bad file specifcation or format"
            exit(0)

        class_documents = put_documents_in_class(documents, document_class)
        if 'category' in class_documents:
            del class_documents['category']
        class_docs, class_terms, vocabulary = get_class_term_counts(class_documents)
        class_terms = transform_to_tfidf(class_terms)
        prior, condProb = train_multinomial_naive_bayes(class_terms, class_docs, vocabulary, alpha)
        for document in testDocuments:
            c = get_document_max_class(prior, condProb, vocabulary, testDocuments[document])
            print document, c
        
def put_documents_in_class(documents, document_class):
    class_documents = defaultdict(lambda: list())

    for document in documents:
        class_documents[document_class[document]].append(documents[document])

    return class_documents

def get_error(testDocuments, test_documents_class, prior, condProb, vocabulary):
    
    classified = 0
    for document in testDocuments:
        c = get_document_max_class(prior, condProb, vocabulary, testDocuments[document])
        if c in test_documents_class[document]:
            classified += 1

    return 1 - (float(classified) / float(len(testDocuments))) #1 because it is the error...

def remove_partition(documents, document_class, current_index, number_folds, id_list):
    step_size = len(id_list)/number_folds
    start = current_index*step_size
    end = (current_index + 1) *step_size
    if number_folds == 1:
        start = 0
        end = 40
    to_remove = id_list[start:end]
    test_documents = dict()
    test_documents_class = dict()
    for key in to_remove:
        if key in documents:
            test_documents[key] = documents[key]
            test_documents_class[key] = document_class[key]
            del documents[key]

    return documents, document_class, test_documents, test_documents_class
        
def run_cross_validation(training_file_path_d, training_file_path_c, alpha):

    test_errors = list()

    documents, document_class = get_training_class_documents_from_file(training_file_path_d, training_file_path_c)
    id_list = list(documents.keys())
    
    for a in alpha:
        print a
        shorter_documents, shorter_document_class, test_documents, test_documents_class = remove_partition(documents, document_class, alpha.index(a), len(alpha), id_list)
        shorter_class_documents = put_documents_in_class(shorter_documents, shorter_document_class)
        if 'category' in shorter_class_documents:
            del shorter_class_documents['category']
        print "put documents in class"
        class_docs, class_terms, vocabulary = get_class_term_counts(shorter_class_documents)
        class_terms = transform_to_tfidf(class_terms)
        print "got term counts"
        prior, condProb = train_multinomial_naive_bayes(class_terms, class_docs, vocabulary, a)
        print "got prior"
        test_error = get_error(test_documents, test_documents_class, prior, condProb, vocabulary)
        test_errors.append(test_error)
        print a, test_error

    print test_errors

   # for e in test_errors:
   #     print alpha[test_errors.index(e)], e
    return alpha[test_errors.index(min(test_errors))], min(test_errors) #returns the best alpha
        
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
    return max(score.iteritems(), key=operator.itemgetter(1))[0]
    
def train_multinomial_naive_bayes(class_terms, class_docs, vocabulary, alpha):
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
            term_count = alpha
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
        class_terms[c] = Counter([porter_stemmer.stem(word.lower()) for word in terms if (word.isalnum() and word not in stopwords.words('english')) ])
        w.update(class_terms[c].iterkeys())

    return class_docs, class_terms, w

def transform_to_tfidf(class_terms):
    term_totals = defaultdict(lambda: 0)
    for c in class_terms:
        for term in class_terms[c]:
            term_totals[term] += class_terms[c][term]
    tfidf = defaultdict(lambda: dict())
    for c in class_terms:
        for term in class_terms[c]:
            value = class_terms[c][term]
            tfidf[c][term] = float(value)/(float(term_totals[term])/4)
            
    return tfidf
def get_training_class_documents_from_file(training_file_path_d, training_file_path_c):
    """Gets docs from csv into python objects
    returns:
    documents - dict(). Key is documentID
    document_class - dict(). Key is documentID.
    """
    documents = dict()
    with open(training_file_path_d, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            documents[line[0]] = line[1]

    document_class = dict()
    with open(training_file_path_c, 'rb') as csvfile1:
        reader2 = csv.reader(csvfile1)
        for line in reader2:
            document_class[line[0]] = line[1]
    return documents, document_class

if __name__ == "__main__":
    main()
