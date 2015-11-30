import networkx as nx
import string
from sys import maxint
import pandas as pd
import numpy as np
import time
import re
import os.path
import math
#num_documents: number of documents
#clean_train_documents: the collection
#unique_words: list of all the words we found 
#sliding_window: window size
#train_par: if true we are in the training documents
#idf_learned
def createGraphFeatures(num_documents,clean_train_documents,unique_words,sliding_window,train_par,idf_learned):
	features = np.zeros((num_documents,len(unique_words)))#where we are going to put the features
	unique_words_len = len(unique_words)
	term_num_docs = {} #dictionay of each word with a count of that word through out the collections
	idf_col = {}#dictionay of each word with the idf of that word
	
	#TO DO:
	#1.idf_col:IDF for the collection
	#	if in training phase compute it
	#	else use the one provided
	#2. term_num_docs : count of the words in the collection
	#	if in training phase populate it
	#	else use the one provided
	if train_par:
		
		#for all documents
		for i in range( 0,num_documents ):
			wordList1 = clean_train_documents[i].split(None)
			wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]
			#count word occurrences through the collection (for idf) put the count in term_num_docs
			if len(wordList2)>1:
				countWords(wordList2,term_num_docs) #TODO: implement this function 
		#TODO: calculate the idf for all words
		for term_x in term_num_docs:
			idf_col[term_x] = math.log10(float(num_documents)/term_num_docs[term_x])            
	# for the testing set
	else:
		#use the existing ones if we are in the test data
		idf_col = idf_learned 
		term_num_docs=unique_words

	print "Creating the graph of words for each document..."
	totalNodes = 0
	totalEdges = 0

	#go over all documents
	for i in range( 0,num_documents ):
		wordList1 = clean_train_documents[i].split(None)
		wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]
		docLen = len(wordList2)
		#the graph
		dG = nx.Graph()

		if len(wordList2)>1:
			populateGraph(wordList2,dG,sliding_window)
			dG.remove_edges_from(dG.selfloop_edges())
			centrality = nx.degree_centrality(dG) #dictionary of centralities (node:degree)

			totalNodes += dG.number_of_nodes()
			totalEdges += dG.number_of_edges()

			
			#TODO : implement comments bellow
			# for all nodes
				#If they are in the desired features
					#compute the TW-IDF score and put it in features[i,unique_words.index(g)]
			for k, node_term in enumerate(dG.nodes()):
				if node_term in idf_col:
					features[i,unique_words.index(node_term)] = centrality[node_term] * idf_col[node_term]

	if train_par:
		nodes_ret=term_num_docs.keys()
		#print "Percentage of features kept:"+str(feature_reduction)
		print "Average number of nodes:"+str(float(totalNodes)/num_documents)
		print "Average number of edges:"+str(float(totalEdges)/num_documents)
	else:
		nodes_ret=term_num_docs
	#return 1: features, 2: idf values (for the test data), 3: the list of terms 
	return features, idf_col, nodes_ret
	
	
def populateGraph(wordList,dG,sliding_window):
	#TODO: implement this function
	#For each position/word in the word list:
		#add the -new- word in the graph
		#for all words -forward- within the window size
			#add new words as new nodes 
			#add edges among all word within the window
	for k, word in enumerate(wordList):
		if not dG.has_node(word):
			dG.add_node(word)
		tempW=sliding_window
		if k+sliding_window > len(wordList):
			tempW=len(wordList) - k
		for j in xrange(1, tempW):
			next_word = wordList[k+j]
			dG.add_edge(word, next_word)
def countWords(wordList,term_num_docs):
	found = set()
	#TODO: implement this function
	#add the terms from the wordlist to the term_num_docs dictionary or increase its count
	for k, word in enumerate(wordList):
		if word not in found:
			found.add(word)
			if word in term_num_docs:
				term_num_docs[word] +=1
			else :
				term_num_docs[word] = 1