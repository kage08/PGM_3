import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re, string
from nltk.stem import WordNetLemmatizer
import logging
from scipy.special import gammaln


def get_corpus(path='data/iitm_train.csv'):

    with open(path, 'r') as fl:
        corpus = fl.readlines()
    logging.debug('Read from '+path)
    logging.debug(''.join(['Sample:',corpus[1]]))
    return corpus[1:]

def pre_process_doc(doc):
    logging.debug("Preprocessing doc")

    #Remove numbers
    doc = re.sub(r'\d+', '', doc)
    
    #Remove Punctuations, whitespaces and lowercase
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    doc = regex.sub('',doc).strip().lower()

    #Tokenize words
    tokens = word_tokenize(doc)

    #Remove stopwords
    tokens = [t for t in tokens if not t in set(stopwords.words('english'))]

    #Remove words with fewer than 3 characters
    tokens = [t for t in tokens if len(t) >= 3]

    #Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens

def get_processed_corpus(path='data/iitm_train.csv'):
    corpus = get_corpus(path)
    return [pre_process_doc(c) for c in corpus]


def dirichlet_denom(alphas):
    alphas = np.array(alphas)
    t1 = gammaln(alphas).sum()
    t2 = gammaln(alphas.sum())
    return t1/t2

