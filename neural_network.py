
# coding: utf-8

# In[2]:

import cPickle
import sklearn 
import numpy as np
import scipy
import csv

def load_sparse_csr(filename):
    loader = np.load(filename)
    return scipy.sparse.csr_matrix(( loader['data'], loader['indices'], loader['indptr']),
                     shape = loader['shape'])
        
def load_csv(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\n')
        array = [float(row[0]) for row in reader]
        return array
    
def load_feature_names(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter= '\n')
        array = [row for row in reader]
        return array


data_train = load_sparse_csr('data_train_best.npz')
data_test = load_sparse_csr('data_test_best.npz')
label_train = load_csv('data/label_train.csv')
label_test = load_csv('data/label_test.csv')
features_names = load_feature_names('data/feature_names.csv')


# In[5]:

label_train = np.array(label_train)



data_traind = data_train.todense()


# In[21]:

from sknn.mlp import Classifier, Layer

nn = Classifier(
    layers=[
        Layer("Tanh", units=50),
        Layer("Softmax")],
        learning_rate=0.0001,
        n_iter=25,
        dropout_rate=0.,
        verbose = True,
        batch_size = 10)
nn.fit(data_train.toarray(), label_train)


# In[22]:

def score (a, b):
    return 1- np.absolute(a - b).mean()
predicted_label = nn.predict(data_test.todense())
print("Neural network - Score on test_data : ", score(predicted_label, label_test))
print("Neural network - Score on train_data : ", score(nn.predict(data_train.todense()), label_train))


scores = []
alphas = np.logspace(-7,0,30)
for alpha in alphas:
    nn = Classifier(
    layers=[
        Layer("Tanh", units=50),
        Layer("Softmax")],
        learning_rate=0.0001,
        n_iter=25,
        dropout_rate=0.,
        verbose = True,
        batch_size = 16)
    nn.fit(data_train.toarray(), label_train)
    predicted_label = nn.predict(data_test.todense())
    scores.append(score(predicted_label, label_test))
    



maxi = scores[0]
amaxi = 0
for i, score in enumerate(scores):
    if maxi< score:
        maxi = score
        amaxi = i
print "max score for i= ", amaxi, "(alpha = ", alphas[amaxi], ")", "performance =", maxi



