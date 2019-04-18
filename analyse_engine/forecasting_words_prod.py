import pylab
#%pylab inline

from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import argparse

import requests
import eventlet
import pandas as pd
import numpy as np
import operator
import mglearn

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

################ <Usage [win example]> ################# 
#python .\forecasting_words_prod.py --view_cluster=yes --max_group_number=80 -cluster=4 --cluster_id=1 --words_number=4

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

# sequence generator from the model
# seed_text - representation of user`s input words sequence
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # pre-pad sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        in_text += ' ' + out_word
    return in_text

#mglearn
def print_topics2(topics, feature_names, sorting, topics_per_chunk=6,
                 n_words=20):
    for i in range(0, len(topics), topics_per_chunk):
        # for each chunk:
        these_topics = topics[i: i + topics_per_chunk]
        # maybe we have less than topics_per_chunk left
        len_this_chunk = len(these_topics)
        # print topic headers
        print(("topic {:<8}" * len_this_chunk).format(*these_topics))
        print(("-------- {0:<5}" * len_this_chunk).format(""))
        # print top n_words frequent words
        for i in range(n_words):           
            try:                
                print(("{:<14}" * len_this_chunk).format(
                    *feature_names[sorting[these_topics, i]]))                
            except:
                pass
        print("\n")
        
def topics_plot(topics, feature_names, sorting, topics_per_chunk=6,
                 n_words=20):
    top_list = []
    for i in range(0, len(topics), topics_per_chunk):
        # for each chunk:
        these_topics = topics[i: i + topics_per_chunk]
        # maybe we have less than topics_per_chunk left
        len_this_chunk = len(these_topics)

        for i in range(n_words):           
            try:                
                top_list.append(("{:1}" * len_this_chunk).format(
                    *feature_names[sorting[these_topics, i]]))                
            except:
                pass
    return top_list

################ <Argparse Injection> #################
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

def words_cluster_type(x):
    x = int(x)
    # for test version
    if x > 4:
        raise argparse.ArgumentTypeError("Max words number for futher prediction is 4")
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
    parser.add_argument ('-cld', '--cluster_id', type=cluster_type, help="Cluster id for learning (word prediction)", required=True)	
    parser.add_argument ('-cluster', type=cluster_type, help="Max cluster id num to test (< 5)", required=True)
    parser.add_argument ('-wn', '--words_number', type=words_cluster_type, help="Words number taken for prediction of the rest line (<=4)", required=True)
    return parser
	
################ <IMDB films extractor> #################
URL = "http://www.imdb.com/chart/top"
r = requests.get(URL)
soup = BeautifulSoup(r.content, 'html.parser')
entries=soup.findAll('div', class_="wlb_ribbon")
# getting top movies ids from imdb
movie_ids = []
for a in entries:
    movie_ids.append(a['data-tconst'])
#print("========= movie_ids:", movie_ids) 
         
header = 'http://www.omdbapi.com/?apikey=6be019fc&tomatoes=true&i='
movie_info = []
for i in movie_ids:
    url = header + i
#print("========= url:", url) 
    r = requests.get(url).json()
    movie = []
    for a in r.keys():
        movie.append(r[a])
    movie_info.append(movie)
columns = r.keys()
df = pd.DataFrame(movie_info, columns = columns)

################ <Extrating from films plot the most valuable words> #################
parser = createParser()
parserArgs = parser.parse_args()
#print('=== mgn = ', parserArgs.max_group_number)
#print('=== parserArgs cl = ', parserArgs.cluster)

plots = list(df['Plot'])
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
# print("=== The first 20 features:\n{}".format(feature_names[:20]))
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
	
################ <Words Prediction part> ################
СLUSTER_ID = parserArgs.cluster_id
WORDS_NUMBER = parserArgs.words_number

print('==== Plot 1 = ', topics_plot(topics=range(СLUSTER_ID), feature_names=feature_names, sorting=sorting, topics_per_chunk=5, n_words=10))
# getting plot_1 line from cluster
plot_1 = topics_plot(topics=range(СLUSTER_ID), feature_names=feature_names, sorting=sorting, topics_per_chunk=5, n_words=10)
plot_string_str = ' '.join([str(x) for x in plot_1])
print('==== plot_string = ', plot_string_str)
plot_string_all = plot_1[:len(plot_1)]
plot_string_input1 = plot_1[: WORDS_NUMBER]
plot_string_input = ' '.join([str(x) for x in plot_string_input1])

tokenizer = Tokenizer()
tokenizer.fit_on_texts([plot_string_all])
encoded = tokenizer.texts_to_sequences([plot_string_all])[0]
vocab_size = len(tokenizer.word_index) + 1

print('Vocabulary Size: %d' % vocab_size)
# encode 2 words -> 1 word
sequences = list()

for i in range(2, len(encoded)):
    sequence = encoded[i-2:i+1]
    sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
# pad sequences
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)
# split into input and output elements
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)

# configure network
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_length-1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network - will print loss and accuracy of each training epoch
model.fit(X, y, epochs=500, verbose=2)

# TEST with PLOT
print('=== PLOT_1 get full line \n', generate_seq(model, tokenizer, max_length-1, plot_string_input, 3))

films_plot_1= []
cluster = generate_seq(model, tokenizer, max_length-1, plot_string_input, 3).split(' ')
print('=== cluster = ', generate_seq(model, tokenizer, max_length-1, plot_string_input, 3).split(' '))
for i in cluster:
    films_plot_1.append(df[df['Plot'].str.contains(i)].Title.unique())

print("========= First cluster for plot 1\n")
for i in set(map(tuple, films_plot_1)):
    print(''.join(list(i)))

# TEST with TEXT
# print(generate_seq(model, tokenizer, max_length-1, 'Jack and', 4))