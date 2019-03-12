import argparse
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from CFModel import CFModel


K_FACTORS = 100 # The number of dimensional embeddings for movies and users

# Function to predict the ratings given User ID and Movie ID
def predict_rating(user_id, movie_id):
    return trained_model.rate(user_id - 1, movie_id - 1)

def give_results(recommended, json_path):
    prepared_json = recommended.to_json(orient='records')

    if json_path:
        with open(json_path, 'w') as outfile:
            json.dump(prepared_json, outfile)
    return prepared_json


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Deep Learning system')
    argparser.add_argument('--data-path',
                           help='Path to folder where stored files: movies.csv, ratings.csv, users.csv',
                           required=True)
    argparser.add_argument('--recommend',
                   help='How many movies to recommend (number)',
                   required=True)
    argparser.add_argument('--user-id',
                   help='Predict films for user',
                   required=True)
    argparser.add_argument('--cache-path',
                   help='Path to the weights.h file (default: {})'.format('./data/weights.h5'),
                   default='./data/weights.h5',
                   required=False)
    argparser.add_argument('--json-path',
                   help='Path to json with recommended movies',
                   required=False)

    argparser.parse_args()
    ratings_path = Path(argparser.parse_args().data_path) / "ratings.csv"
    users_path = Path(argparser.parse_args().data_path) / "users.csv"
    movies_path = Path(argparser.parse_args().data_path) / "movies.csv"
    recommend = argparser.parse_args().recommend
    cache_path = argparser.parse_args().cache_path
    recommend = int(argparser.parse_args().recommend)
    interesting_user = int(argparser.parse_args().user_id)
    json_path = argparser.parse_args().json_path


    ###LOAD DATASETS
    # Reading ratings file
    ratings = pd.read_csv(str(ratings_path), sep='\t', encoding='latin-1', 
                          usecols=['user_id', 'movie_id', 'user_emb_id', 'movie_emb_id', 'rating'])
    max_userid = ratings['user_id'].drop_duplicates().max()
    max_movieid = ratings['movie_id'].drop_duplicates().max()

    # Reading ratings file
    users = pd.read_csv(str(users_path), sep='\t', encoding='latin-1', 
                        usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

    # Reading ratings file
    movies = pd.read_csv(str(movies_path), sep='\t', encoding='latin-1', 
                         usecols=['movie_id', 'title', 'genres'])

    ###CREATING SETS
    # Create training set
    shuffled_ratings = ratings.sample(frac=1., random_state=42)
    # Shuffling users
    Users = shuffled_ratings['user_emb_id'].values
    #print('Users:', Users, ', shape =', Users.shape)
    # Shuffling movies
    Movies = shuffled_ratings['movie_emb_id'].values
    #print ('Movies:', Movies, ', shape =', Movies.shape)
    # Shuffling ratings
    Ratings = shuffled_ratings['rating'].values
    #print ('Ratings:', Ratings, ', shape =', Ratings.shape)

    ###PREDICT THE RATING
    # The next step is to actually predict the ratings a random user will give to a random movie.
    # Below I apply the freshly trained deep learning model for all the users and all the movies,
    # using 100 dimensional embeddings for each of them.
    # I also load pre-trained weights from weights.h5 for the model.
    # Use the pre-trained model
    trained_model = CFModel(max_userid, max_movieid, K_FACTORS)
    # Load weights
    trained_model.load_weights(cache_path)
    # Pick a random test user
    users[users['user_id'] == interesting_user]

    #Show the top n movies that user 2000 has already rated, including the predictions column showing the values that used 2000 would have rated based on the defined predict_rating function.
    user_ratings = ratings[ratings['user_id'] == interesting_user][['user_id', 'movie_id', 'rating']]

    ###RECOMMEND MOVIES
    #List of unrated n movies sorted by prediction value for our test user.
    recommendations = ratings[ratings['movie_id'].isin(user_ratings['movie_id']) == False][['movie_id']].drop_duplicates()
    recommendations['prediction'] = recommendations.apply(lambda x: predict_rating(interesting_user, x['movie_id']), axis=1)
    recommended = recommendations.sort_values(by='prediction',
                                              ascending=False).merge(movies,
                                                                     on='movie_id',
                                                                     how='inner',
                                                                     suffixes=['_u', '_m']).head(recommend)
    print(recommended.to_string(header=False))

    print(give_results(recommended, json_path))
