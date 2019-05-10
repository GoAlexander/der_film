import pylab
#%pylab inline

import requests
import eventlet
import pandas as pd
import operator
#pip install mglearn
import mglearn
import numpy as np
import argparse
import json

from bs4 import BeautifulSoup
from sklearn.decomposition import PCA
from IMDBAPI import IMDB
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation

imdb = IMDB()

################ <Functions> ################# 
def plotcounts (a):
    temp=a
    temp.replace(".", " ").replace(",", "")
    temp=temp.split(" ")
    temp=map(lambda x:x.lower(),temp)
    words=[]
    for b in movie_words:
        if b in temp:
            words.append(b)
    return words

def worddummy (a):
    if m in a:
        return 1
    else:
        return 0

def countactors(a):
    temp.append(a.split(", "))
    return temp

def token(text):
    return(text.split("|"))

def group_type(x):
    x = int(x)
    if x > 84:
        raise argparse.ArgumentTypeError("Maximum group number is 84")
    return x

def cluster_type(x):
    x = int(x)
    # for test version
    if x > 5:
        raise argparse.ArgumentTypeError("Max cluster id num to test is 5")
    return x

def strBool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')	
	
def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-view', '--view_cluster', type=strBool, help="Display chosen cluster", required=True)	
    parser.add_argument ('-mgn', '--max_group_number', type=group_type, help="Display max number of cluster groups (< 84)", required=True)
    parser.add_argument ('-cluster', type=cluster_type, help="Max cluster id num to test (< 5)", required=True)
    return parser

################ <IMDB Json data processing> #################
#url = "https://github.com/GoAlexander/der_film/blob/master/analyse_engine/data/Movie_imdb_26539.json"
#[26540 rows x 15 columns]
df = pd.read_json('data/Movie_imdb_26539.json', orient='columns', encoding='utf8')

################ <Extrating from films plot the most valuable words> #################
parser = createParser()
parserArgs = parser.parse_args()
#print('=== mgn = ', parserArgs.max_group_number)
#print('=== parserArgs cl = ', parserArgs.cluster)

plots = list(df['keywords'].apply(', '.join))
temp = ""
for p in plots:
	temp = temp + p
temp.replace(".", " ").replace(",", "")
temp = temp.split(" ")
temp = list(map(lambda x:x.lower(),temp))
#print("=== plot_temp:", temp)

# collect valuable for further groups generatig words
vect = CountVectorizer(max_features=10000, max_df=.15, stop_words="english").fit(temp)
X = vect.transform(temp)

# test result
feature_names = vect.get_feature_names()
# print("=== Number of features: {}".format(len(feature_names)))
print("=== The first 20 features:\n{}".format(feature_names[:20]))
# print("=== Each 2000 feature:\n{}".format(feature_names[::2000]))

# Max and min groups number (if number is greater than MAX_GROUPS_NUMBER - groups won`t be dense, info become useless)
# Here is LDA (Latent Dirichlet allocation - Латентное размещение Дирихле) is used for clustering (building topic model)
MAX_GROUPS_NUMBER = parserArgs.max_group_number
#MIN_GROUPS_NUMBER = 10
lda = LatentDirichletAllocation(n_components=MAX_GROUPS_NUMBER, learning_method="batch", max_iter=25, random_state=0)
document_topics = lda.fit_transform(X)

# Sorting of features acsending order 
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
# Getting of features names
feature_names = np.array(vect.get_feature_names())
# Take some number of topics to view results
if parserArgs.view_cluster:
	mglearn.tools.print_topics(topics=range(20), feature_names=feature_names,
	 sorting=sorting, topics_per_chunk=5, n_words=10)

################ <Demo: taking films from first two groups> ################# 
# (!!!) Take into account: films duplications were removed
# sorting by weight document of the topic number 1
group = np.argsort(document_topics[:, parserArgs.cluster])[::-1]
films = []

for i in group[:10]:
    films.append(df[df['Plot'].str.contains(temp[i])].Title.unique())

print("========= %d cluster\n" % parserArgs.cluster)
for i in set(map(tuple, films)):
    print(''.join(list(i)))
       
# sorting by weight document of the topic number 3
# group3 = np.argsort(document_topics[:, 3])[::-1]
# films_3 = []

# for i in group3[:10]:
#     films_3.append(df[df['Plot'].str.contains(temp[i])].Title.unique())

# print("========= Third cluster\n")
# for i in set(map(tuple, films_3)):
#     print('\n', list(i))
    
# and so on ...
# printing will be optimized in one method