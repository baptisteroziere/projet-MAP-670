import random
from pyspark import SparkContext
from functools import partial
import string
import loadLabeled from loadFiles
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import networkx as nx
import csv
from scipy.sparse import coo_matrix

def tw(doc_words, sliding_window, idf_col):
    dG = nx.Graph()
    if len(doc_words)>1:
        populateGraph(doc_words,dG,sliding_window)
        dG.remove_edges_from(dG.selfloop_edges())
        centrality = nx.degree_centrality(dG)
        tw_list = [(term, centrality[term] * idf_col[term]) for term in dG.nodes() if term in idf_col]
    return tw_list

def populateGraph(wordList,dG,sliding_window):
    for k, word in enumerate(wordList):
        if not dG.has_node(word):
            dG.add_node(word)
        if k+sliding_window > len(wordList):
            tempW=len(wordList) - k
        else:
            tempW=sliding_window
        for j in xrange(1, tempW):
            next_word = wordList[k+j]
            dG.add_edge(word, next_word)

stemmer = SnowballStemmer("english")

class tw_idf():
    def __init__(self, sliding_window, num_documents):
        self.sliding_window = sliding_window
        self.num_documents = num_documents
    def fit(self, cleaned_rdd):
        unique_words = cleaned_rdd.flatMap(lambda x: x.split()).distinct().collect()
        self.unique_words = {word: i for i, word in enumerate(unique_words)}
        word_counts = cleaned_rdd.flatMap(lambda x: x.split())
        word_counts = word_counts.map(lambda word : (word, 1)).reduceByKey(lambda x, y: x + y)
        idf = word_counts.map(lambda x: (x[0], np.log10(float(self.num_documents)/x[1]))).collect()
        self.idf_col = {x : y for x,y in idf}
        return self
    def transform(self, cleaned_rdd):
        words_per_doc=cleaned_rdd.map(lambda x: x.split())
        tw_scores = words_per_doc.map(partial(tw, sliding_window=self.sliding_window, idf_col=self.idf_col)).collect()
        row=[]
        col=[]
        val=[]
        for i in range(len(tw_scores)):
            row += len(tw_scores[i])*[i]
            for j in range(len(tw_scores[i])):
                col += [self.unique_words[tw_scores[i][j][0]]]
                val += [tw_scores[i][j][1]]
        row=np.array(row)
        col=np.array(col)
        val=np.array(val)
        return coo_matrix((val,(row,col)), shape=(len(tw_scores),len(self.unique_words)))

path_baptiste = "/home/baptiste/Documents/data/train"
path_igor = "C:/Users/Igor/Documents/Master Data Science/Big Data Analytics/Projet/data/train"
path_sofia = "/Users/Flukmacdesof/data 2/train"
path_sofia_server = "/home/sofia.calcagno/data 2/train"

#TRAIN

print "importing data..."

data, labels = loadLabeled(path_sofia_server)
sliding_window = 2
num_documents = len(data)
sc = SparkContext(appName="TW-IDF App")

print "starting data cleaning"

exclude = set(string.punctuation) #punctuation list
swords = stopwords.words("english") #list of stopwords

rdd=sc.parallelize(data, numSlices=16)
cleaned_rdd=rdd.map(lambda x: x.replace('<br />',' ')).map(lambda x:''.join([stemmer.stem(word)+' ' for word in x.split() if word not in swords])).map(lambda x:''.join([w if w not in exclude else " " for w in x.lower() ]))

TW = tw_idf(sliding_window, num_documents).fit(cleaned_rdd)
m_tw_idf_train = TW.transform(cleaned_rdd)
save_sparse_csr("tw_idf_train",tw_idf_matrix)
