import random
from pyspark import SparkContext
from functools import partial
from features_creation_TWIDF_Sofia import loadLabeled
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import networkx as nx
import csv

def save_sparse_csr(filename, array):
    np.savez(filename, data = array.data, indices = array.indices, 
             indptr = array.indptr, shape = array.shape)

def tw(doc_words):
    dG = nx.Graph()
    if len(doc_words)>1:
        populateGraph(doc_words,dG,sliding_window)
        dG.remove_edges_from(dG.selfloop_edges())
        centrality = nx.degree_centrality(dG)
        tw = [(term, centrality[term] * idf_col[term]) for term in dG.nodes() if term in idf_col]
    return tw

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

path_baptiste = "/home/baptiste/Documents/data/train"
path_igor = "C:/Users/Igor/Documents/Master Data Science/Big Data Analytics/Projet/data/train"
path_sofia = "/Users/Flukmacdesof/data 2/train"
path_sofia_server = "/home/sofia.calcagno/data 2/train"

print "importing data..."

data, labels = loadLabeled(path_sofia_server)


num_documents = len(data)
sliding_window = 2

sc = SparkContext(appName="TW-IDF App")

print "starting data cleaning"

exclude = set(string.punctuation) #punctuation list                        
swords = stopwords.words("english") #list of stopwords

rdd=sc.parallelize(data, numSlices=16)
#cleaning the data
cleaned_rdd=rdd.map(lambda x: x.replace('<br />',' ')).map(lambda x:''.join([stemmer.stem(word)+' ' for word in x.split() if word not in swords])).map(lambda x:''.join([w if w not in exclude else " " for w in x.lower() ])) 

#words, unique words and word count
words = cleaned_rdd.flatMap(lambda x: x.split(" ")).filter(lambda word: word not in ["", " "])
words_per_doc=cleaned_rdd.map(lambda x: x.split(" "))
unique_words=words.distinct()
unique_words_list=unique_words.collect()
word_counts = words.map(lambda word : (unique_words_list.index(word), 1)).reduceByKey(lambda x, y: x + y).filter(lambda x: x[0] not in [" ",""])


#TW-IDF computation
#IDF
idf = word_counts.map(lambda x: (x[0], np.log10(float(num_documents)/x[1])))
idf_tuples = idf.collect()
idf_col = dict((x,y) for x,y in idf_tuples)
idf_name=idf_col.keys()

tw_scores = words_per_doc.map(tw)
features_tw=tw_scores.collect()

row=[]
col=[]
val=[]
for i in range(len(features_tw)):
    row += len(features_tw[i])*[i]
    for j in range(len(features_tw[i])):
        col += [features_tw[i][j][0]]
        val += [features_tw[i][j][1]]
row=np.array(row)
col=np.array(col)
val=np.array(val)

tw_idf_matrix=coo_matrix((val,(row,col)), shape=(len(features_tw),len(len(unique_words_list))))
save_sparse_csr("tw_idf_train",tw_idf_matrix)
