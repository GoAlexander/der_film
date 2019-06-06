import pylab

import requests
import eventlet
import pandas as pd
import operator
#pip install mglearn
import mglearn
import numpy as np
import argparse
import json
import os
import pickle
import nltk

#nltk.download('sentiwordnet')
#from nltk.corpus import sentiwordnet as swn 

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
from gensim.models.ldamulticore import LdaMulticore

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

def str_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')	
	
def create_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument ('-view', '--view_cluster', type=strBool, help="Display chosen cluster", required=True)	
    #parser.add_argument ('-mgn', '--max_group_number', type=group_type, help="Display max number of cluster groups (< 84)", required=True)
    #parser.add_argument ('-cluster', type=cluster_type, help="Max cluster id num to test (< 5)", required=True)
    parser.add_argument ('-words', help="Words for films search (<= 4)",  default=[], nargs=4, required=True)
    return parser


def cached(cachefile):
    def decorator(fn):  # define a decorator for a function "fn"
        def wrapped(*args, **kwargs):   # define a wrapper that will finally call "fn" with all arguments            
			# if cache exists -> load it and return its content
            if os.path.exists(cachefile):
                    with open(cachefile, 'rb') as cachehandle:
                        print("using cached result from '%s'" % cachefile)
                        return pickle.load(cachehandle)

            # execute the function with all arguments passed		
            res = fn(*args, **kwargs)
            # write to cache file
            with open(cachefile, 'wb') as cachehandle:
                print("saving result to cache '%s'" % cachefile)
                pickle.dump(res, cachehandle)
            return res
        return wrapped
    return decorator

@cached('film_dataframe_cache.res')	
def get_film_dataframe():	
	df = pd.read_csv('../data/movie_metadata.csv')
	df = df.dropna(subset=['plot_keywords'])
	df['plot_keywords'] = df['plot_keywords'].str.replace('|', ' ')
	return df
	
@cached('preprocessed_data_cache.res')
def data_preprocessing(df):
	plots = list(df['plot_keywords'])
	temp = ""
	for p in plots:
		temp = temp + p + " "
	temp.replace(",", "")
	temp = temp.split(" ")
	return list(map(lambda x:x.lower(),temp))
	
def get_lda(MAX_GROUPS_NUMBER, temp):
	vect = CountVectorizer(max_df=.15, stop_words="english").fit(temp)
	feature_names = vect.get_feature_names()
	#print("=== Number of features: {}".format(len(feature_names)))
	#print("=== The first 20 features:\n{}".format(feature_names[:20]))
	X = vect.transform(temp)
	return LatentDirichletAllocation(n_components=MAX_GROUPS_NUMBER, learning_method="batch", max_iter=50, random_state=0)

@cached('lda_transform_cache.res')
def get_transformed_data(MAX_GROUPS_NUMBER, df):
    lda = get_lda(MAX_GROUPS_NUMBER, df)
    return lda.fit_transform(X)

@cached('dataframe_cache.res')
def get_df_res(datasetArray):
    datasetArray = datasetArray.applymap(str)
    for col in datasetArray:
        for i, row_value in datasetArray[col].iteritems():
            index = pd.to_numeric(datasetArray[col][i])
            datasetArray.at[i, col] = temp[index]
        datasetArray.drop_duplicates(subset=col, inplace=True)
    return datasetArray
	
################ <IMDB data processing> #################
#df = pd.read_csv('../data/movie_metadata.csv')

################ <Extrating from films plot the most valuable words> #################
parser = create_parser()
parserArgs = parser.parse_args()
#print('=== mgn = ', parserArgs.max_group_number)
#print('=== parserArgs cl = ', parserArgs.cluster)
#print('=== words = ', parserArgs.words)

df = get_film_dataframe()
temp = data_preprocessing(df)

# Max and min groups number (if number is greater than MAX_GROUPS_NUMBER - groups won`t be dense, info become useless)
# Here is LDA (Latent Dirichlet allocation - Латентное размещение Дирихле) is used for clustering (building topic model)
MAX_GROUPS_NUMBER = 83 #parserArgs.max_group_number
#MIN_GROUPS_NUMBER = 10

#83 columns, 37226 rows
document_topics = get_transformed_data(MAX_GROUPS_NUMBER, df)

# Sorting of features acsending order 
#sorting = np.argsort(lda.components_, axis=1)[:, ::-1]

# Getting of features names
#feature_names = np.array(vect.get_feature_names())
#print("========= feature_names \n", feature_names)

# Take some number of topics to view results
#if parserArgs.view_cluster:
#	mglearn.tools.print_topics(topics=range(20), feature_names=feature_names,
#	sorting=sorting, topics_per_chunk=5, n_words=10)

################ <Demo: taking films from first two groups> ################# 
# (!!!) Take into account: films duplications were removed
# sorting by weight document of the topic number 1

# 37226 len of group
#for i in MAX_GROUPS_NUMBER:
# document_topics[:, parserArgs.cluster] contains weight of words
#print("====== document_topics[:, parserArgs.cluster] = ", document_topics[:, parserArgs.cluster])
datasetArray = pd.DataFrame(np.argsort(document_topics, axis=0)[::-1])

datasetArrayRes = get_df_res(datasetArray)
#print("====== datasetArrayRes", datasetArrayRes)

films = []
data = {}

wordsList = parserArgs.words
print("=====wordsList= ", wordsList)

for col in datasetArrayRes:
    #print("====== Cluster number = ", col)
    if not datasetArrayRes[datasetArrayRes[col].str.contains('|'.join(wordsList), case=False)].empty:
        res = datasetArrayRes[col][datasetArrayRes[col].str.contains('|'.join(wordsList), case=False)].values
        for i in res:
            films.append(df[df['plot_keywords'].str.contains(i)].movie_title.unique())
			
for i in set(map(tuple, films)):
	data['film_name'] = list(i)
	print('\n'.join(list(i)))

#json_data = json.dumps(data)
#print("===== json_data = ", json_data)

with open('data.json', 'w') as outfile:
	json.dump(data, outfile, ensure_ascii=False)