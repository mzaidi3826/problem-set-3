'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd
import ast

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    model_pred_df = pd.read_csv("prediction_model_03.csv")
    genres_df = pd.read_csv("genres.csv")
    return model_pred_df, genres_df


def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''
    genre_list = genres_df['genre'].tolist()

    genre_true_counts = {}
    genre_tp_counts = {}
    genre_fp_counts = {}

    for _, row in model_pred_df.iterrows():
        try:
            true_genres = set(ast.literal_eval(row['actual genres']))
        except:
            true_genres = set()

        predicted_genres = {row['predicted'].strip()} if pd.notna(row['predicted']) else set()

        # filter to valid genres
        true_genres = {g for g in true_genres if g in genre_list}
        predicted_genres = {g for g in predicted_genres if g in genre_list}

        for genre in true_genres:
            # true count
            if genre not in genre_true_counts:
                genre_true_counts[genre] = 0
            genre_true_counts[genre] += 1

            # true positive count
            if genre in predicted_genres:
                if genre not in genre_tp_counts:
                    genre_tp_counts[genre] = 0
                genre_tp_counts[genre] += 1

        for genre in predicted_genres:
            if genre not in true_genres:
                if genre not in genre_fp_counts:
                    genre_fp_counts[genre] = 0
                genre_fp_counts[genre] += 1

    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts