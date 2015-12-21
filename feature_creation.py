from loadFiles import loadLabeled
from tw_fc_functions import *
from other_fc_functions import *
import numpy as np
from sklearn.cross_validation import train_test_split

path_baptiste_train = "/home/baptiste/Documents/data/train"
path_igor_train = "C:/Users/Igor/Documents/Master Data Science/Big Data Analytics/Projet/data/train"
path_sofia_train = "/Users/Flukmacdesof/data 2/train"
path_sofia_server_train = "/home/sofia.calcagno/data 2/train"

path_baptiste_test = "/home/baptiste/Documents/data/test"
path_igor_test = "C:/Users/Igor/Documents/Master Data Science/Big Data Analytics/Projet/data/test"
path_sofia_test = "/Users/Flukmacdesof/data 2/test"
path_sofia_server_test = "/home/sofia.calcagno/data 2/test"

exclude = set(string.punctuation) #punctuation list
swords = stopwords.words("english") #list of stopwords

#whole train data
train, train_labels = loadLabeled(path_sofia_server_train)
#whole test data
test, names = loadUknown(path_sofia_server_test)
#train divided into train / test data
train_train, train_test, train_train_label, train_test_label = train_test_split(
data, labels, test_size = 0.25, random_state = 42
)

sc = SparkContext(appName="TW-IDF App")

rdd_train=sc.parallelize(train, numSlices=16)
cleaned_rdd=rdd_train.map(lambda x: x.replace('<br />',' ')).map(
lambda x:''.join(
[stemmer.stem(word)+' ' for word in x.split() if word not in swords]
)
).map(
lambda x:''.join(
[w if w not in exclude else " " for w in x.lower() ]
)
)
