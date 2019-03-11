#%matplotlib inline
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from CFModel import CFModel


# Define constants
K_FACTORS = 100 # The number of dimensional embeddings for movies and users
TEST_USER = 2000 # A random test user (user_id = 2000)


# Function to predict the ratings given User ID and Movie ID
def predict_rating(user_id, movie_id):
    #return trained_model.rate(user_id - 1, movie_id - 1)
    return model.rate(user_id - 1, movie_id - 1)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Deep Learning system')
    argparser.add_argument('--ratings',
                           help='Path to ratings csv',
                           required=True)
    argparser.add_argument('--users',
                       help='Path to users csv',
                       required=True)
    argparser.add_argument('--movies',
                   help='Path to movies csv',
                   required=True)
    argparser.add_argument('--recommend',
                   help='How many movies to recommend (number)',
                   required=True)

    argparser.parse_args()

    ratings = argparser.parse_args().ratings
    users = argparser.parse_args().users
    movies = argparser.parse_args().movies
    recommend = argparser.parse_args().recommend


    ###LOAD DATASETS
    # Reading ratings file
    ratings = pd.read_csv('ratings.csv', sep='\t', encoding='latin-1', 
                          usecols=['user_id', 'movie_id', 'user_emb_id', 'movie_emb_id', 'rating'])
    max_userid = ratings['user_id'].drop_duplicates().max()
    max_movieid = ratings['movie_id'].drop_duplicates().max()

    # Reading ratings file
    users = pd.read_csv('users.csv', sep='\t', encoding='latin-1', 
                        usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

    # Reading ratings file
    movies = pd.read_csv('movies.csv', sep='\t', encoding='latin-1', 
                         usecols=['movie_id', 'title', 'genres'])


    ###CREATING SETS
    # Create training set
    shuffled_ratings = ratings.sample(frac=1., random_state=42)

    # Shuffling users
    Users = shuffled_ratings['user_emb_id'].values
    print('Users:', Users, ', shape =', Users.shape)

    # Shuffling movies
    Movies = shuffled_ratings['movie_emb_id'].values
    print ('Movies:', Movies, ', shape =', Movies.shape)

    # Shuffling ratings
    Ratings = shuffled_ratings['rating'].values
    print ('Ratings:', Ratings, ', shape =', Ratings.shape)


    ###DEEP LEARNING MODEL
    #Compile the model using Mean Squared Error (MSE) as the loss function and the AdaMax learning algorithm.
    # Define model
    model = CFModel(max_userid, max_movieid, K_FACTORS)
    # Compile the model using MSE as the loss function and the AdaMax learning algorithm
    model.compile(loss='mse', optimizer='adamax')


    ###TRAIN the MODEL
    # Callbacks monitor the validation loss
    # Save the model weights each time the validation loss has improved
    callbacks = [EarlyStopping('val_loss', patience=2), 
                 ModelCheckpoint('weights.h5', save_best_only=True)]

    # Use 30 epochs, 90% training data, 10% validation data 
    history = model.fit([Users, Movies], Ratings, nb_epoch=10, validation_split=.1, verbose=2, callbacks=callbacks)


    ###ROOT MEAN SQUARE  ERROR
    #During the training process above, I saved the model weights each time the validation loss has improved. Thus, I can use that value to calculate the best validation Root Mean Square Error.
    # Show the best validation RMSE
    min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
    print('Minimum RMSE at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))


    ###PREDICT THE RATING
    #The next step is to actually predict the ratings a random user will give to a random movie. Below I apply the freshly trained deep learning model for all the users and all the movies, using 100 dimensional embeddings for each of them. I also load pre-trained weights from weights.h5 for the model.
    # Use the pre-trained model
    trained_model = CFModel(max_userid, max_movieid, K_FACTORS)
    # Load weights
    trained_model.load_weights('weights.h5')
    # Pick a random test user
    users[users['user_id'] == TEST_USER]

    #Show the top 20 movies that user 2000 has already rated, including the predictions column showing the values that used 2000 would have rated based on the defined predict_rating function.
    user_ratings = ratings[ratings['user_id'] == TEST_USER][['user_id', 'movie_id', 'rating']]
    user_ratings['prediction'] = user_ratings.apply(lambda x: predict_rating(TEST_USER, x['movie_id']), axis=1)
    user_ratings.sort_values(by='rating', 
                             ascending=False).merge(movies, 
                                                    on='movie_id', 
                                                    how='inner', 
                                                    suffixes=['_u', '_m']).head(20)


    ###RECOMMEND MOVIES
    #List of unrated 20 movies sorted by prediction value for our test user.
    recommendations = ratings[ratings['movie_id'].isin(user_ratings['movie_id']) == False][['movie_id']].drop_duplicates()
    recommendations['prediction'] = recommendations.apply(lambda x: predict_rating(TEST_USER, x['movie_id']), axis=1)
    recommended = recommendations.sort_values(by='prediction',
                                              ascending=False).merge(movies,
                                                                     on='movie_id',
                                                                     how='inner',
                                                                     suffixes=['_u', '_m']).head(recommend)
    print("Result, recommended movies:")
    print(recommended)
