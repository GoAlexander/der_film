import argparse
import math
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from CFModel import CFModel


# Define constants
K_FACTORS = 100 # The number of dimensional embeddings for movies and users
LOSS_FACTOR = 'mse'
OPTIMIZER = 'adamax'
NUMBER_EPOCH = 10


# Function to predict the ratings given User ID and Movie ID
def predict_rating(user_id, movie_id):
    #return trained_model.rate(user_id - 1, movie_id - 1)
    return model.rate(user_id - 1, movie_id - 1)

def dump_usage_statistics(input, rmse):
    new_data = {rmse: input_vars}

    with open('film_learner_dump.json') as f:
        data = json.load(f)
    data.update(new_data)
    with open('film_learner_dump.json', 'w') as f:
        json.dump(data, f)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Deep Learning system. Learner.')
    argparser.add_argument('--data-path',
                           help='Path to folder where stored files: movies.csv, ratings.csv, users.csv',
                           required=True)
    argparser.add_argument('--recommend',
                   help='How many movies to recommend (number)',
                   required=True)

    argparser.parse_args()

    ratings_path = Path(argparser.parse_args().data_path) / "ratings.csv"
    users_path = Path(argparser.parse_args().data_path) / "users.csv"
    movies_path = Path(argparser.parse_args().data_path) / "movies.csv"
    recommend = argparser.parse_args().recommend


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
    model.compile(loss=LOSS_FACTOR, optimizer=OPTIMIZER)


    ###TRAIN the MODEL
    # Callbacks monitor the validation loss
    # Save the model weights each time the validation loss has improved
    callbacks = [EarlyStopping('val_loss', patience=2), # TODO: patience?
                 ModelCheckpoint('weights.h5', save_best_only=True)]

    # Use 30 epochs, 90% training data, 10% validation data 
    history = model.fit([Users, Movies], Ratings, nb_epoch=NUMBER_EPOCH, validation_split=.1, verbose=2, callbacks=callbacks)


    ###ROOT MEAN SQUARE  ERROR
    #During the training process above, I saved the model weights each time the validation loss has improved. Thus, I can use that value to calculate the best validation Root Mean Square Error.
    # Show the best validation RMSE
    min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
    min_rmse = '{:.4f}'.format(math.sqrt(min_val_loss))
    print('Minimum RMSE at epoch', '{:d}'.format(idx+1), '=', min_rmse)

    input_vars = [LOSS_FACTOR, OPTIMIZER, K_FACTORS, NUMBER_EPOCH]
    dump_usage_statistics(input_vars, min_rmse)
