from loadFiles import *
from tw_fc_functions import *
from other_fc_functions import *
import numpy as np
from sklearn.cross_validation import train_test_split
from nltk import PorterStemmer
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from scipy.sparse import hstack, coo_matrix, csr_matrix
from pyspark.mllib.feature import StandardScaler
import functools

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

def rdd_to_sparse(rdd, unique_words):
    collected = rdd.collect()
    row=[]
    col=[]
    val=[]
    for i in range(len(collected)):
        row += len(collected[i])*[i]
        for j in range(len(collected[i])):
            col += [unique_words[collected[i][j][0]]]
            val += [collected[i][j][1]]
    row=np.array(row)
    col=np.array(col)
    val=np.array(val)
    return coo_matrix((val,(row,col)), shape=(len(collected),len(unique_words)))

def create_features_tw(train, train_labels, test, test_size=0.25,
compute_together=True, compute_other=True,
sliding_window=2, scale=True):
    """
    Give a function that splits the data for cross-validation that only takes
    train and train_labels as arguments (use partial).
    If you want to compute the idf in all the available data,
    use compute_together=True,
    if you want to respect the train / test split use Falses
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
        if scale == True:
            scaler=StandardScaler()
            fit_scale=[scaler.fit(transform_tw[0]), scaler.fit(transform_tw[2])]
            transform_tw=[
            fit_scale[0].transform(transform_tw[0]),
            fit_scale[0].transform(transform_tw[1]),
            fit_scale[1].transform(transform_tw[2]),
            fit_scale[1].transform(transform_tw[3]),
            ]
        list_transform_tw = map(lambda x: x.collect(), transform_tw)
        to_sparse_funcs=[functools.partial(
        rdd_to_sparse, unique_words=fit_tw[0].unique_words),
        functools.partial(rdd_to_sparse, unique_words=fit_tw[1].unique_words)
        ]
        sparse_tw=[to_sparse_funcs[0](list_transform_tw[0]),
        to_sparse_funcs[0](list_transform_tw[1]),
        to_sparse_funcs[1](list_transform_tw[2]),
        to_sparse_funcs[1](list_transform_tw[3])
        ]
        features = [hstack([sparse_tw[i], other[i][0]]) for i in range(len(rdd_list))]
        names = [fit_tw[0].idf_col.keys() + other[0][1],
        fit_tw[0].idf_col.keys() + other[1][1],
        fit_tw[1].idf_col.keys() + other[2][1],
        fit_tw[1].idf_col.keys() + other[3][1]]
        return features, names
    else :
        all_data= train + test
        rdd_all=sc.parallelize(all_data, numSlices=16)
        no_html_all_rdd=rdd_all.map(lambda x: x.replace('<br />',' '))
        other_all, other_names =create_other_features(no_html_all_rdd)
        rdd_all=rdd_all.map(
        lambda x: x.lower()
        ).map(lambda
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
        num_documents=len(train)
        tw_fit=tw_idf(num_documents=num_documents,
        sliding_window=sliding_window).fit(rdd_all)
        tw=tw_fit.transform(rdd_all)
        if scale ==True:
            scaler=StandardScaler()
            tw=scaler.fit(tw).transform(tw)
        collect_tw =tw.collect()
        to_sparse_func=functools.partial(
        rdd_to_sparse, unique_words=tw_fit.unique_words)
        sparse_tw=to_sparse_func(collect_tw)
        features = hstack([sparse_tw, other_all[0]])
        names = tw_fit.idf_col.keys() + other_names
        return features, names

file_names_separate=["train", "test", "train_train", "train_test"]
for sliding_window in range(1,6):
    separate_tw=create_features_tw(train, train_labels, test, test_size=0.25,
    compute_together=False, compute_other=True,
    sliding_window=sliding_window, scale=True)
    for i, split in enumerate(separate_tw[0]):
        np.savez(
        "tw_sw{}_{}".format(sliding_window,file_names_separate[i]),
        data = split.data,
        row=split.row,
        col=split.col,
        shape = split.shape
        )

for sliding_window in range(1,6):
    joined_tw = create_features_tw(train, train_labels, test, test_size=0.25,
    compute_together=True, compute_other=True,
    sliding_window=sliding_window, scale=True)
    data_train = joined_tw[0].tocsr()[:25000]
    data_test= joined_tw[0].tocsr()[25000:]
    data_train_train, data_train_test=train_test_split(
    data_train, test_size = 0.25, random_state = 42)
    np.savez(
    "tw_sw{}_all_train".format(sliding_window),
    data = data_train.data,
    indices = data_train.indices,
    indptr = data_train.indptr,
    shape = data_train.shape
    )
    np.savez(
    "tw_sw{}_all_test".format(sliding_window),
    data = data_test.data,
    indices = data_test.indices,
    indptr = data_test.indptr,
    shape = data_test.shape
    )
    np.savez(
    "tw_sw{}_all_train_train".format(sliding_window),
    data = data_train_train.data,
    indices = data_train_train.indices,
    indptr = data_train_train.indptr,
    shape = data_train_train.shape
    )
    np.savez(
    "tw_sw{}_all_train_test".format(sliding_window),
    data = data_train_test.data,
    indices = data_train_test.indices,
    indptr = data_train_test.indptr,
    shape = data_train_test.shape
    )
