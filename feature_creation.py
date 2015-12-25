from loadFiles import *
from tw_fc_functions import *
from other_fc_functions import *
import numpy as np
from sklearn.cross_validation import train_test_split
from nltk import PorterStemmer
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from scipy.sparse import hstack, coo_matrix, csr_matrix

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
sliding_window=2, n_grams=None):
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
            num_documents=len(train)
            fit_tw=[
            tw_idf(
            num_documents=num_documents, sliding_window=sliding_window).fit(
            rdd_list[0]),
            tw_idf(
            num_documents=num_documents, sliding_window=sliding_window
            ).fit(rdd_list[2])]
            transform_tw = [fit_tw[0].transform(rdd_list[0]),
            fit_tw[0].transform(rdd_list[1]), fit_tw[1].transform(rdd_list[2]),
            fit_tw[1].transform(rdd_list[3])]
            features = [hstack([transform_tw[i], other[i][0]]) for i in range(len(rdd_list))]
            names = [fit_tw[0].idf_col.keys() + other[0][1],
            fit_tw[0].idf_col.keys() + other[1][1],
            fit_tw[1].idf_col.keys() + other[2][1],
            fit_tw[1].idf_col.keys() + other[3][1]]
            return features, names
        elif word_score == "tf_idf":
            hashingTF = HashingTF()
            tf_rdd = [hashingTF.transform(rdd) for rdd in rdd_list]
            tfidf=[]
            for tf in tf_rdd:
                tf.cache()
                idf = IDF().fit(tf)
                tfidf.append(idf.transform(tf))

            to_sparse=[]
            lengths=[]
            for k in range(len(tfidf)):
                to_sparse.append(tfidf[k].map(lambda x: (x.indices, x.values)))
                lengths.append(tfidf[k].map(lambda x: len(x.values)).collect())

            col_val_array=[to_sparse[k].reduce(
            lambda x, y: np.hstack([x, y])) for k in range(len(to_sparse))]


            rows = []
            for  k in range(len(to_sparse)):
                rows_temp=[]
                for i in range(len(lengths[k])):
                    rows_temp+=[i]*lengths[k][i]
                rows.append(rows_temp)

            idx_col_val=[np.concatenate([np.array([rows[k]]),col_val_array[k]]) for k in range(len(to_sparse))]
            tfidf_matricmatrices = [
            coo_matrix((idx_col_val[k][2],
            (idx_col_val[k][0],idx_col_val[k][1]))) for k in range(len(tfidf))
            return features, names
    else :
        all_data= train + test
        rdd_all=sc.parallelize(all_data, numSlices=16)
        no_html_all_rdd=rdd_all.map(lambda x: x.replace('<br />',' '))
        other_all, other_names =create_other_features(no_html_all_rdd)
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
            num_documents=len(train)
            tw_fit=tw_idf(num_documents=num_documents,
            sliding_window=sliding_window).fit(rdd_all)
            tw=tw_fit.transform(rdd_all)
            features = hstack([tw, other_all])
            names = fit_tw.idf_col.keys() + other_names
            return features, names
        elif word_score == "tf_idf":
            hashingTF = HashingTF()
            tf_rdd = hashingTF.transform(rdd_all)
            tf.cache()
            idf = IDF().fit(tf)
            tfidf=idf.transform(tf).collect()
            #Besoin de transformer tfidf en sparse matrix...
            #features =
            #names =
            return features, names

        rdd_all= sc.parallelize(train + test, numSlices=16)








test_size=0.25
compute_together=True
word_score="tw_idf"
compute_other=True
sliding_window=2

]
