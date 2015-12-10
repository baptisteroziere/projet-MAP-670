import random
from pyspark import SparkContext
from functools import partial
from features_creation_TWIDF_Sofia import *

path_baptiste = "/home/baptiste/Documents/data/train"
path_igor = "C:/Users/Igor/Documents/Master Data Science/Big Data Analytics/Projet/data/train"
path_sofia = "/Users/Flukmacdesof/data 2/train"
path_sofia_server = "/home/sofia.calcagno/data 2/train"

print "importing data..."

data, labels = loadLabeled(path_sofia)


num_documents = len(data)
sliding_window = 2
train_par = True
idf_learned = {}

print "starting Spark Context"

sc = SparkContext(appName="TW-IDF App")

print "starting data cleaning"

rdd=sc.parallelize(data, numSlices=16).map(data_preparation_twidf)
unique_words=rdd.flatMap(lambda x: x.split("" )).distinct()
print "successfully cleaned data"

print "starting graph"

graph_data=sc.parallelize(rdd,numSlices=16).map(partial(createGraphFeatures,num_documents=num_documents, unique_words=unique_words,sliding_window=sliding_window,train_par=train_par,idf_learned=idf_learned)).collect()

graph_data.saveAsTextFile(tw_idf_features)

