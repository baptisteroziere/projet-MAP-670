import os
import numpy as np
import codecs
from nltk.stem.snowball import SnowballStemmer
#from imdbpie import Imdb
import string
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix

#various features
happy = [":-)", ":)", ":D", ":o)", ":]", ":3", ":c)", ":>", "=]", "8)", "=)",
":}", ":^)", ":-))", "^^"]
laughing = [":-D", "8-D", "8D", "x-D", "xD", "X-D", "XD", "=-D", "=D", "=-3",
"=3", "B^D"]
sad = [">:[", ":-(", ":(", ":-c", ":c", ":-<", ":<", ":-[", ":[", ":{", ";("]
cry = [":'-(", ":'("]
happy_cry = [":'-)", ":')"]
horror = ["D:<", "D:", "D8", "D;", "D=", "DX", "v.v", "D-':"]
surprised = [">:O", ":-O", ":O", ":-o", ":o", "8-0", "O_O", "o-o", "O_o", "o_O",
"o_o", "O-O"]
kiss= [":*", ":^*", "( '}{' )"]
wink = [";-)", ";)", "*-)", "*)", ";-]", ";]", ";D", ";^)", ":-"]
tongue = [">:P", ":-P", ":P", "X-P", "x-p", "xp", "XP", ":-p", ":p", "=p",
":-b", ":b", "d:"]
skeptical = [">:\ ".replace(" ", ""), ">:/",
":-/", ":-.", ":/", ":\ ".replace(" ", ""),
 "=/", "=\ ".replace(" ", ""), ":L", "=L", ":S", ">.<"]
neutral = [":|", ":-|"]
angel = ["O:-)", "0:-3", "0:3", "0:-)", "0:)", "0;^)"]
evil = [">:)", ">;)", ">:-)", "}:-)", "}:)", "3:-)", "3:)"]
high_five = ["o/\o", "^5", ">_>^ ^<_<"]
heart = ["<3"]
broken_hart = ["</3"]
angry = [":@"]
smiley_list = [
happy, laughing, sad, cry, happy_cry, horror, surprised, kiss, wink,
tongue, skeptical, neutral, angel, evil, high_five, heart, broken_hart, angry]
smiley_names = [ "happy", "laughing", "sad", "cry", "happy_cry", "horror",
"surprised", "kiss", "wink", "tongue", "skeptical", "neutral", "angel", "evil",
"high_five", "heart", "broken_hart",  "angry" ]


# Get anything being before " /10" or "over ten" or between both
def get_all_mentionned_grade(review):
    rev_grade = []
    rev_grade.append([])
    review_split= review.split()
    for idw, word in enumerate(review_split):
        try:
            if((word=="on" or word=="over") and idw+1<len(review_split)):
                if(review_split[idw+1][0:3]=="ten" or review_split[idw+1][0:2]=="10"):
                    if(idw>0):
                         rev_grade.append(review_split[idw-1])
        except: print review_split
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
                            rev_grade.append(word[0:idx])
                        else:
                            if(idw>0):
                                rev_grade.append(review_split[idw-1])
    return rev_grade


# Convert what have been collected to a grade. If impossible, then the grade is 5.4321
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
                elif g_new[0] in '.,':
                    new_grade = g_new[1:]
                else:
                    new_grade = g_new
        elif (grade[-1] not in good) and (grade.lower() in numerical_words):
            new_grade = numerical_words[grade.lower()]
        return float(new_grade)


# Same as previous but in the case that the reviews mentionned many grades
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


# Return a single grade for each review
def creation_grade(review):
    rev_grade = get_all_mentionned_grade(review)
    new_rev_grade = []
    for idg, grade in enumerate(rev_grade):
        converted_grade = 5.4321
        if grade != []:
            if len(grade) == 1:
                converted_grade = convert_to_real_grade(grade[0])
            else:
                converted_grade = convert_to_real_grade_2(grade)
        new_rev_grade.append(converted_grade)
    return new_rev_grade

# return two OneHotEncoder (Or dummies)
# The first says if it is a good movie
# The second says if it is a bad movie
def creation_good_grade_bad_grade(grades):
    good_grade = []
    bad_grade = []
    for grade in grades:
        if grade > 6.8:
            good_grade.append(True)
            bad_grade.append(False)
        elif grade <4.0:
            good_grade.append(False)
            bad_grade.append(True)
        else:
            good_grade.append(False)
            bad_grade.append(False)
        return good_grade, bad_grade

def good_and_bad_movies():
    good_movies = {}
    bad_movies = {}
    imdb = Imdb(anonymize = True)
    for movie in imdb.top_250():
        good_movies[movie['title']] = movie['rating']
    return good_movies, bad_movies

def mentionned_movies(review):
    good_movies, bad_movies = good_and_bad_movies()
    wrong_titles = wrong_titles = ['M', 'Up', 'Ran']
    rev_movies = []
    # Movie in the review
    movies = []
    for key, value in good_movies.items():
        if key in review and key not in wrong_titles:
            movies.append(value)
    for key, value in bad_movies.items():
        if key in review and key not in wrong_titles:
            movies.append(value)
    rev_movies.append(movies)
    return rev_movies

def creation_good_bad_mentionned_movies(review):
    list_mentionned_movies = mentionned_movies(review)
    good_movie_mentionned = []
    bad_movie_mentionned = []
    for movie in list_mentionned_movies:
        if movie > 6.8:
            good_movie_mentionned.append(True)
            bad_movie_mentionned.append(False)
        elif movie < 4.0:
            good_movie_mentionned.append(False)
            bad_movie_mentionned.append(True)
        else:
            good_movie_mentionned.append(False)
            bad_movie_mentionned.append(False)
    return good_movie_mentionned, bad_movie_mentionned

def good_and_bad_movies():
    good_movies = {}
    bad_movies = {}
    imdb = Imdb(anonymize = True)
    for movie in imdb.top_250():
        good_movies[movie['title']] = movie['rating']
    return good_movies, bad_movies

def mentionned_movies(review):
    good_movies, bad_movies = good_and_bad_movies()
    wrong_titles = wrong_titles = ['M', 'Up', 'Ran']
    rev_movies = []
    movies = []
    for key, value in good_movies.items():
        if key in review and key not in wrong_titles:
            movies.append(value)
    for key, value in bad_movies.items():
        if key in review and key not in wrong_titles:
            movies.append(value)
    rev_movies.append(movies)
    return rev_movies

def creation_good_bad_mentionned_movies(review):
    list_mentionned_movies = mentionned_movies(review)
    good_movie_mentionned = []
    bad_movie_mentionned = []
    for movie in list_mentionned_movies:
        if movie > 6.8:
            good_movie_mentionned.append(True)
            bad_movie_mentionned.append(False)
        elif movie < 4.0:
            good_movie_mentionned.append(False)
            bad_movie_mentionned.append(True)
        else:
            good_movie_mentionned.append(False)
            bad_movie_mentionned.append(False)
    return good_movie_mentionned, bad_movie_mentionned

def create_other_features(no_html_rdd):
    rdd_split = no_html_rdd.map(lambda x: x.split())
    wordcount = rdd_split.map(lambda x: len(x)).collect()
    excla = no_html_rdd.map(lambda x: x.count('!')).collect()
    inter = no_html_rdd.map(lambda x: x.count('?')).collect()
    length = no_html_rdd.map(lambda x: len(x)).collect()
    susp = no_html_rdd.map(lambda x: x.count('...')).collect()
    smiley_ind=[rdd_split.map(
    lambda x: 1 if len(
    set(x).intersection(set(smileys))
    )>0 else 0).collect() for smileys in smiley_list]
    #good_bad_movies = [no_html_rdd.map(creation_good_bad_mentionned_movies).collect()]
    grades_int = no_html_rdd.map(creation_good_grade_bad_grade)
    good_grades = no_html_rdd.map(lambda x: 1 if x[0] else 0).collect()
    bad_grades = no_html_rdd.map(lambda x: 1 if x[1] else 0).collect()
    #missing good_bad_movies
    features_list = [
    wordcount, excla, inter, length, susp
    ] + smiley_ind + [good_grades, bad_grades]
    features = np.array(features_list).T
    #missing good_bad_movies
    names = [
    "wordcount", "excla", "inter", "length", "susp"] + smiley_names + [
    "good_grades", "bad_grades"]
    return csr_matrix(features), names
