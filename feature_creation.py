from loadFiles import *
from tw_fc_functions import *
from other_fc_functions import *
import numpy as np
from sklearn.cross_validation import train_test_split
from nltk import PorterStemmer
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

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
stemmer=PorterStemmer()

#whole train data
train, train_labels = loadLabeled(path_sofia_server_train)
#whole test data
test, names = loadUnknown(path_sofia_server_test)
#train divided into train / test data

random_state=42
sc = SparkContext(appName="TW-IDF App")

def create_features(train, train_labels, test, test_size=0.25,
compute_together=True, word_score="tw_idf", compute_other=True,
sliding_window=None, n_grams=None):
    """
    Give a function that splits the data for cross-validation that only takes
    train and train_labels as arguments (use partial).
    If you want to compute the idf in all the available data,
    use compute_together=True,
    if you want to respect the train / test split use False
    Choose your word_score : "tw_idf" or "tf_idf".
    Choose whether to include custom features (punctuation, document length...
    or not setting compute_other to True or False)
    """
    if not compute_together:
        train_train, train_test, train_train_label, train_test_label = train_test_split(
        train, train_labels, test_size=test_size, random_state=random_state)
        rdd_train=sc.parallelize(train, numSlices=16).map(
        lambda x: x.replace('<br />',' ')
        )
        rdd_test=sc.parallelize(test, numSlices=16).map(
        lambda x: x.replace('<br />',' ')
        )
        rdd_train_train=sc.parallelize(train_train, numSlices=16).map(
        lambda x: x.replace('<br />',' ')
        )
        rdd_train_test=sc.parallelize(train_test, numSlices=16).map(
        lambda x: x.replace('<br />',' ')
        )
        rdd_list=[rdd_train, rdd_test, rdd_train_train, rdd_train_test]
        other=[create_other_features(rdd) for rdd in rdd_list]
        #clean data
        for i, rdd in enumerate(rdd_list):
            rdd_list[i]=rdd.map(
            lambda x: x.lower()
            ).map(
            lambda x: x.replace(
            "'ll", " will"
            ).replace(
            "n't", " not"
            ).replace(
            "i'm", "i am"
            ).replace(
            "you're", "you are"
            )).map(
            lambda x:' '.join(
            [stemmer.stem(word) for word in x.split() if word not in swords]
            )
            ).map(
            lambda x:''.join(
            [w if w not in exclude else " " for w in x.lower() ]
            )
            )
        if word_score == "tw_idf":
            fit_tw=[tw_idf().fit(rdd_list[0]), tw_idf().fit(rdd_list[2])]
            transform_tw = [fit_tw[0].transform(rdd_list[0]),
            fit_tw[0].transform(rdd_list[1]), fit_tw[1].transform(rdd_list[2]),
            fit_tw[1].transform(rdd_list[3])]
            features = [csr_matrix(np.array(transform_tw[i] + other[i][0])) for i in range(len(rdd_list))]
            names = [fit_tw[i].idf_col.keys() + other[i][1] for i in range(len(rdd_list))]
            return features, names
        elif word_score == "tw_idf":
            hashingTF = HashingTF()
            tf_rdd = [hashingTF.transform(rdd) for rdd in rdd_list]
            tfidf=[]
            for i, tf in enumerate(tf_rdd):
                tf.cache()
                idf = IDF().fit(tf)
                tfidf[i] = idf.transform(tf).collect()
            features = [csr_matrix(np.array(tfidf[i] + other[i][0])) for i in range(len(rdd_list))]
            names = [fit_tw[i].idf_col.keys() + other[i][1] for i in range(len(rdd_list))]
            return features, names
    else :
        all_data= train + test
        rdd_all=sc.parallelize(all_data, numSlices=16)
        no_html_all_rdd=rdd_all.map(lambda x: x.replace('<br />',' '))
        other_all, names =create_other_features(no_html_all_rdd)
        rdd_all=rdd_all.map(
        lambda x: x.lower()
        ).map(
        x: x.replace(
        "'ll", " will"
        ).replace(
        "n't", " not"
        ).replace(
        "i'm", "i am"
        ).replace(
        "you're", "you are"
        )).map(
        lambda x:' '.join(
        [stemmer.stem(word) for word in x.split() if word not in swords]
        )
        ).map(
        lambda x:''.join(
        [w if w not in exclude else " " for w in x.lower() ]
        )
        )
        if word_score == "tw_idf":
            tw=tw_idf().fit(rdd_all).transform(rdd_all)
        elif word_score == "tf_idf":
            hashingTF = HashingTF()
            tf_rdd = hashingTF.transform(rdd)
            tf.cache()
            idf = IDF().fit(tf).transform(tf)

        rdd_all= sc.parallelize(train + test, numSlices=16)



rdd.map(lambda x: x.replace('<br />',' '))



.map(
lambda x:''.join(
[stemmer.stem(word)+' ' for word in x.split() if word not in swords]
)
).map(
lambda x:''.join(
[w if w not in exclude else " " for w in x.lower() ]
)
)

if not compute_together:
    train_train, train_test, train_train_label, train_test_label = train_test_split(
    train, train_labels, test_size=test_size, random_state=random_state)
    rdd_train=sc.parallelize(train, numSlices=16).map(
    lambda x: x.replace('<br />',' ')
    )
    rdd_test=sc.parallelize(test, numSlices=16).map(
    lambda x: x.replace('<br />',' ')
    )
    rdd_train_train=sc.parallelize(train_train, numSlices=16).map(
    lambda x: x.replace('<br />',' ')
    )
    rdd_train_test=sc.parallelize(train_test, numSlices=16).map(
    lambda x: x.replace('<br />',' ')
    )
    rdd_list=[rdd_train, rdd_test, rdd_train_train, rdd_train_test]
    other=[create_other_features(rdd) for rdd in rdd_list]
    #clean data
    for i, rdd in enumerate(rdd_list):
        rdd_list[i]=rdd.map(
        lambda x: x.lower()
        ).map(
        lambda x: x.replace(
        "'ll", " will"
        ).replace(
        "n't", " not"
        ).replace(
        "i'm", "i am"
        ).replace(
        "you're", "you are"
        )).map(
        lambda x:' '.join(
        [stemmer.stem(word) for word in x.split() if word not in swords]
        )
        ).map(
        lambda x:''.join(
        [w if w not in exclude else " " for w in x.lower() ]
        )
        )
    if word_score == "tw_idf":
        fit_tw=[tw_idf().fit(rdd_list[0]), tw_idf().fit(rdd_list[2])]
        transform_tw = [fit_tw[0].transform(rdd_list[0]),
        fit_tw[0].transform(rdd_list[1]), fit_tw[1].transform(rdd_list[2]),
        fit_tw[1].transform(rdd_list[3])]
        features = [csr_matrix(np.array(transform_tw[i] + other[i][0])) for i in range(len(rdd_list))]
        names = [fit_tw[i].idf_col.keys() + other[i][1] for i in range(len(rdd_list))]
        #return features, names
    elif word_score == "tw_idf":
        hashingTF = HashingTF()
        tf_rdd = [hashingTF.transform(rdd) for rdd in rdd_list]
        tfidf=[]
        for i, tf in enumerate(tf_rdd):
            tf.cache()
            idf = IDF().fit(tf)
            tfidf[i] = idf.transform(tf).collect()
        features = [csr_matrix(np.array(tfidf[i] + other[i][0])) for i in range(len(rdd_list))]
        names = [fit_tw[i].idf_col.keys() + other[i][1] for i in range(len(rdd_list))]
