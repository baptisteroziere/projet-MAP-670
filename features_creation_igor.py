
# coding: utf-8


# In[1]:
import os
import numpy as np
import codecs
path_baptiste = "/home/baptiste/Documents/data/train"
path_igor = "C:/Users/Igor/Documents/Master Data Science/Big Data Analytics/Projet/Data/train"
path_sofia = "/Users/Flukmacdesof/data 2/train"
path_serveur = "/home/igor.koval/projet-MAP-670/data/train"


#assumes labelled data ra stored into a positive and negative folder
#returns two lists one with the text per file and another with the corresponding class 
def loadLabeled(path):

	rootdirPOS =path+'/pos'
	rootdirNEG =path+'/neg'
	data=[]
	Class=[]
	count=0
	for subdir, dirs, files in os.walk(rootdirPOS):
		
		for file in files:
			with codecs.open(rootdirPOS+"/"+file, 'r',encoding="utf-8") as content_file:
				content = content_file.read() #assume that there are NO "new line characters"
				data.append(content)
	tmpc1=np.ones(len(data))
	for subdir, dirs, files in os.walk(rootdirNEG):
		
		for file in files:
			with codecs.open(rootdirNEG+"/"+file, 'r',encoding="utf-8") as content_file:
				content = content_file.read() #assume that there are NO "new line characters"
				data.append(content)
	tmpc0=np.zeros(len(data)-len(tmpc1))
	Class=np.concatenate((tmpc1,tmpc0),axis=0)
	return data,Class
#loads unlabelled data	
#returns two lists
#one with the data per file and another with the respective filenames (without the file extension)
def loadUknown(path):
	rootdir=path
	data=[]
	names=[]
	for subdir, dirs, files in os.walk(rootdir):
		for file in files:
			with open(rootdir+"/"+file, 'r', encoding= "utf-8") as content_file:
				content = content_file.read() #assume that there are NO "new line characters"
				data.append(content)
				names.append(file.split(".")[0])
	return data,names


# ## Data loading

# In[2]:
#sc = SparkContext(appName="Simple App")
reviews, Class = loadLabeled(path_serveur)
print np.asarray(reviews).shape


# ## First data cleaning:
# - Remove all the HTML symbols

# In[3]:

# Remove HLML signs
HTMLlist = ['<br />']

for idx, review in enumerate(reviews):
    for word in HTMLlist:
        reviews[idx] = review.replace(word,' ')


# ## Feature creation:
# - List punctuation (various form)

# In[4]:

excla = [0]*len(reviews)
inter = [0]*len(reviews)
susp = [0]*len(reviews)
for i, review in enumerate(reviews):
    for char in review:
        if char == "?":
            inter[i] += 1
        elif char == "!":
            excla[i] += 1


# In[5]:

from nltk.tokenize.casual import TweetTokenizer
ttoken = TweetTokenizer(reduce_len=True)
tokenized_reviews = []

for review in reviews:
    tokenized_reviews.append(ttoken.tokenize(review))
    
for i, review in enumerate(tokenized_reviews):
    for word in review:
        if word == "...":
            susp[i] += 1


# ## Feature creation : 
# - Length of the review
# - Movie mentionned in the review

# In[6]:

rev_length = []
rev_word_count = []


for idr,review in enumerate(reviews):
    # Length of the review
    rev_length.append(len(review))
    
    words = 0
    for word in review:
        words +=1
    rev_word_count.append(words)


# In[7]:

# THE MOVIE LIST HAS TO BE COMPLETED
'''
from imdbpie import Imdb
imdb = Imdb(anonymize = True)

# Movie list creation
movie_list = {}
for movie in imdb.top_250():
    movie_list[movie['title']] = movie['rating']

# In[8]:

rev_movie = []
wrong_titles = ['M', 'Up', 'Ran']

for idr,review in enumerate(reviews):   
    # Movie in the review
    movies = []
    for key, value in movie_list.items():
        if key in review and key not in wrong_titles:
            movies.append(value)
    rev_movie.append(movies)


# In[9]:

# TO BE DISCUSSED : WHAT IF MORE THAN 1 MOVIE MENTIONNED?
new_rev_movie = []

for movie in rev_movie:
    # No movies quoted or more than 1 movie quoted
    if len(movie) != 1:
        new_rev_movie.append(-5.4321)
    
    # Only one movie quoted
    else:
        new_rev_movie.append(movie[0])     

# In[10]:

# AS FOR NOW, THERE ARE ONLY BAD MOVIES
good_movie_mentionned = []
bad_movie_mentionned = []

for movie in new_rev_movie:
    if movie > 6.8:
        good_movie_mentionned.append(True)
        bad_movie_mentionned.append(False)
    elif movie < 4.0:
        good_movie_mentionned.append(False)
        bad_movie_mentionned.append(True)
    else:
        good_movie_mentionned.append(False)
        bad_movie_mentionned.append(False)


# ## Feature Creation :
# - Grade mentionned in the movie
'''
# In[11]:

rev_grade = []
# Grade/Mark in the review
for idr,review in enumerate(reviews):
    rev_grade.append([])
    review_split= review.split(" ")

    for idw, word in enumerate(review_split):
        try:
            if((word=="on" or word=="over") and idw+1<len(review_split)):
                if(review_split[idw+1][0:3]=="ten" or review_split[idw+1][0:2]=="10"):
                    if(idw>0):
                         rev_grade[idr].append(review_split[idw-1])
        except: print(review_split)
        
        for idx, char in enumerate(word):
                if char == '/':
                    ten_is_there= False
                    if(idx < len(word)-2):
                        if word[idx+1] == '1' and word[idx+2] == '0':
                            ten_is_there=True
                    if(idx < len(word)-3):
                        if word[idx+1] == 't' and word[idx+2] == 'e' and word[idx+3]=="n":
                            ten_is_there=True
                    if(idx== len(word) -1 and idw<len(review_split)-1 and len(review_split[idw+1])>1 ):
                        if((review_split[idw+1][0]=='1' and review_split[idw+1][1]=='0') or review_split[idw+1][0:3]=="ten"):
                            ten_is_there=True
                        
                    if(ten_is_there):                  
                        if(idx)>0:
                            rev_grade[idr].append(word[0:idx])
                        else:
                            if(idw>0):
                                rev_grade[idr].append(review_split[idw-1])
        


# In[12]:

# DO NOT DELETE THE PRINT NOW : The function may be better when used on the test set

# SOMETIMES : 1/10/2015 -> It is a date ! 

def convert_to_real_grade(grade):
    new_grade = 5.4321
    
    ### The grade is a float
    try:
        new_grade = float(grade)
        return float(new_grade)
    
    ### The grade is not a float
    except:
        good = '0123465789'
        numerical_words = {'zero':0, 'one':1, 'two':1, 'three':3, 'four':4, 'five':5, 
                           'six':6, 'seven':7, 'height':8, 'nine':9, 'ten':10}
        
        ## The grade has numerical values at the end 
        if grade[-1] in good:

            ### Read the grade in the string
            one_dot = False
            g_new = ''
            for char in reversed(grade):
                if char in good:
                    g_new = char + g_new
                elif char in '.,' and one_dot == False:
                    g_new  = '.' + g_new
                    one_dot = True
                else:
                    if g_new[0] in '.,':
                        new_grade = g_new[1:]
                    else:
                        new_grade = g_new
                    
        elif (grade[-1] not in good) and (grade.lower() in numerical_words):
            new_grade = numerical_words[grade.lower()]
            
        return float(new_grade)
                
                    

# functions for reviews with more than one grade
def convert_to_real_grade_2(grade):
    final_grade = 5.4321
    new_grades = []
    for i in range(len(grade)):
        new_grade = convert_to_real_grade(grade[i])
        new_grades.append(new_grade)
        
    if final_grade not in new_grades:
        ## NEXT CONDITION IS TO BE DISCUSSED
        if np.max(new_grades) - np.min(new_grades) < 7:
            final_grade = np.mean(new_grades)
    return final_grade


# In[13]:

new_rev_grade = []
for idg, grade in enumerate(rev_grade):    
    converted_grade = 5.4321
    
    if grade != []:
        if len(grade) == 1:
            converted_grade = convert_to_real_grade(grade[0])
        else:
            converted_grade = convert_to_real_grade_2(grade)
    new_rev_grade.append(converted_grade)


# In[14]:

#Dummies
good_grade = []
bad_grade = []

for grade in new_rev_grade:
    if grade > 6.8:
        good_grade.append(True)
        bad_grade.append(False)
    elif grade <4.0:
        good_grade.append(False)
        bad_grade.append(True)
    else:
        good_grade.append(False)
        bad_grade.append(False)


# ## Creating new features : Other ideas to try
# 
# - Find N-grams where it may start with a CAPITAL (As for the movie names and actor's names)
# - Add smileys

# In[15]:

happy = [":-)", ":)", ":D", ":o)", ":]", ":3", ":c)", ":>", "=]", "8)", "=)", ":}", ":^)", ":-))", "^^"]
laughing = [":-D", "8-D", "8D", "x-D", "xD", "X-D", "XD", "=-D", "=D", "=-3", "=3", "B^D"]
sad = [">:[", ":-(", ":(", ":-c", ":c", ":-<", ":<", ":-[", ":[", ":{", ";("]
cry = [":'-(", ":'("]
happy_cry = [":'-)", ":')"]
horror = ["D:<", "D:", "D8", "D;", "D=", "DX", "v.v", "D-':"]
surprised = [">:O", ":-O", ":O", ":-o", ":o", "8-0", "O_O", "o-o", "O_o", "o_O", "o_o", "O-O"]
kiss= [":*", ":^*", "( '}{' )"]
wink = [";-)", ";)", "*-)", "*)", ";-]", ";]", ";D", ";^)", ":-"]
tongue = [">:P", ":-P", ":P", "X-P", "x-p", "xp", "XP", ":-p", ":p", "=p", ":-b", ":b", "d:"]
skeptical = [">:\ ".replace(" ", ""), ">:/", ":-/", ":-.", ":/", ":\ ".replace(" ", ""), "=/", "=\ ".replace(" ", ""), ":L", "=L", ":S", ">.<"]
neutral = [":|", ":-|"]
angel = ["O:-)", "0:-3", "0:3", "0:-)", "0:)", "0;^)"]
evil = [">:)", ">;)", ">:-)", "}:-)", "}:)", "3:-)", "3:)"]
high_five = ["o/\o", "^5", ">_>^ ^<_<"]
heart = ["<3"]
broken_hart = ["</3"]
angry = [":@"]
smiley_list = [
happy,
laughing,
sad,
cry,
happy_cry,
horror,
surprised,
kiss,
wink,
tongue,
skeptical,
neutral,
angel,
evil,
high_five,
heart,
broken_hart, 
angry]
smiley_names = [
"happy",
"laughing",
"sad",
"cry",
"happy_cry",
"horror",
"surprised",
"kiss",
"wink",
"tongue",
"skeptical",
"neutral",
"angel",
"evil",
"high_five",
"heart",
"broken_hart", 
"angry"
]


# In[16]:

def gen_features_smiley(tokenized_text, smiley_list):
    features_smiley = np.zeros((len(tokenized_text),len(smiley_list)))
    for i, review in enumerate(tokenized_text):
        for w in review :
            if len(w)<2 : 
                pass
            elif len(w)>5:
                pass
            for j, cat in enumerate(smiley_list):
                if w in cat:
                    features_smiley[i,j] = 1
    return features_smiley


# In[17]:

features_smiley = gen_features_smiley(tokenized_reviews, smiley_list)


# ## Second Data Cleaning (After the features creation) :
#  - Punctuation
#  - Stop Words
# 

# In[18]:

# Remove punctuation, lower all characters
# exclude = {',' ,'+', '<', ':', '/', ']', '(', ')', '{', '"', '_', '?', '@', '}', ...}
import string
exclude = set(string.punctuation)
for idx, review in enumerate(reviews):
    reviews[idx] = ''.join([w if w not in exclude else " " for w in review.lower() ])
    
# Remove stop words based on the given list - To be changed depending on the needs
from nltk.corpus import stopwords
stopwords = stopwords.words("english")
for idx, review in enumerate(reviews):
    reviews[idx] = ''.join([w +' ' for w in review.split() if w not in stopwords])


# ## Third Data Cleaning: 
# - Stemmisation

# In[19]:

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


# In[20]:

# Steeming -> Reduce words to their initial mining

for idx, review in enumerate(reviews):
    reviews[idx] = ''.join([stemmer.stem(word)+' ' for word in review.split()])


# ## Tf - Idf Matrix
# 
# #### To be upgraded with new tf and idf functions!

# In[21]:

# Features extraction with TF - IDF : get the matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

m = TfidfVectorizer()
tfidf_matrix = m.fit_transform(reviews)

print("Size of the tfidf matrix: ", tfidf_matrix.size)
print(tfidf_matrix.shape)


# ## Adding the new features to the Tf-Idf matrix
# #### New features are : 
# - Number of exclamation point
# - Number of interrogation point
# - Number of suspension point
# - Review length
# - Number of word (word_count)
# - Movie mentionned
# - Grade mentionned
# 
# - Smileys (How to deal with them?)

# In[22]:

import scipy

def csr_vappend(a,b): #b est un vecteur ligne (np.array ou liste) et a est une sparse matrix
    if(type(a)!= scipy.sparse.csr.csr_matrix):
        a=scipy.sparse.csr_matrix(a)
        
    if(type(b)== list):
        b=np.array([b]).T
    if(type(b)!= scipy.sparse.csr.csr_matrix):
        b=scipy.sparse.csr_matrix(b)
       
    a=scipy.sparse.hstack([a,b])
    


# In[23]:

csr_vappend(tfidf_matrix, excla)
csr_vappend(tfidf_matrix, inter)
csr_vappend(tfidf_matrix, susp)
csr_vappend(tfidf_matrix, rev_length)
csr_vappend(tfidf_matrix, rev_word_count)
csr_vappend(tfidf_matrix, good_grade)
csr_vappend(tfidf_matrix, bad_grade)
#csr_vappend(tfidf_matrix, good_movie_mentionned)
#csr_vappend(tfidf_matrix, bad_movie_mentionned)
csr_vappend(tfidf_matrix, features_smiley)


# ##  CSV Creation:
# 
# Create 5 csv : train_train.csv, train_test.csv, y_train_train.csv, y_train_test.csv, test.csv

# In[24]:

# Split the tf-idf matrix into two data sets to process the cross validation : training and test set
from sklearn.cross_validation import train_test_split

data_train, data_test, label_train, label_test = train_test_split(tfidf_matrix, Class, test_size = 0.3, random_state = 42)


# ##### 

# In[25]:

import csv

def save_sparse_csr(filename, array):
    np.savez(filename, data = array.data, indices = array.indices, 
             indptr = array.indptr, shape = array.shape)
    
def load_sparse_csr(filename):
    loader = np.load(filename)
    return scipy.sparse.csr_matrix(( loader['data'], loader['indices'], loader['indptr']),
                     shape = loader['shape'])

def save_csv(filename, array):
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter = '\n')
        writer.writerow(array)
        
def load_csv(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\n')
        array = [float(row[0]) for row in reader]
        return array


# In[26]:

save_sparse_csr('data_train', data_train)
save_sparse_csr('data_test', data_test)
save_csv('label_train.csv', label_train)
save_csv('label_test.csv', label_test)


# In[ ]:



