import numpy as np
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise.evaluate import evaluate
import ast
class CollaborativeFilter :

    def __init__(self) :

        #Trains svd algorithm to be able to predict ratings for a user in our small ratings dataset
        pd.options.mode.chained_assignment = None

        ratings_path = './data/ratings_small.csv'
        md_path = './data/movies_metadata.csv'
        links_path = './data/links_small.csv'

        md = pd.read_csv(md_path, low_memory=False,
                         usecols=['genres', 'id', 'overview', 'production_companies', 'status', 'tagline', 'title',
                                  'vote_average', 'vote_count'],
                         dtype={'tagline': str, 'overview': str}, na_values={'vote_average': 0, 'vote_count': 0,
                                                                             'status': ['Canceled', 'In Production',
                                                                                        'Planned', 'Rumored']})
        md.drop(labels=[19730, 29503, 35587], axis=0, inplace=True)
        md.dropna(subset=['status'], axis=0, inplace=True)
        md['id'] = md['id'].astype(int)

        links = pd.read_csv(links_path)
        links.dropna(axis=0, subset=['tmdbId'], inplace=True)

        ids = links['tmdbId'].astype(int)
        self.smd = md[md['id'].isin(ids)]
        self.smd.reset_index(drop=True, inplace=True)
        self.smd['genres'] = self.smd['genres'].apply(
            lambda g_row: list(map(lambda g_item: g_item['name'].strip().lower(), ast.literal_eval(g_row))))

        self.ratings = pd.read_csv(ratings_path)

        reader = Reader(rating_scale=(1,5))

        dataset = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']],reader)

        dataset.split(n_folds=5)

        self.svd = SVD()

        #evaluate(svd,dataset,measures=['rmse','mae'])

        trainset = dataset.build_full_trainset()

        self.svd.fit(trainset)

        print("Recommender ready.")

    def recommend(self,user_id,genre=None,num_recommends=25) :
        # Input: user_id
        # Output: list of num_recommends (25) recommended movies for that user, based on svd algorithm

        if user_id not in self.ratings['userId'].unique() :
            print("User:",user_id,"not in dataset. Please select another user.")
            return

        movies = self.smd
        #Filter out movies the user has already rated/watched
        rated_movie_ids = self.ratings[self.ratings['userId'] == user_id]['movieId']
        qualified_movies = movies[~movies['id'].isin(rated_movie_ids)]
        if type(genre) is not str :
            print("Invalid genre. Genre must be string")
            return
        if genre :
            for i, movie in qualified_movies.iterrows():
                if genre.lower() not in movie['genres']:
                    qualified_movies.drop(labels=i, axis=0, inplace=True)
        if qualified_movies.empty :
            print("Invalid genre")
            return

        predictions = [self.svd.predict(user_id, m_id) for m_id in qualified_movies['id'].values]
        qualified_movies['est_rate'] = 0.0
        qualified_movies['est_rate'] = [est for uid, iid, r_true, est, _ in predictions]
        recommendations = qualified_movies.sort_values(by='est_rate', ascending=False).loc[:,['title','overview','genres','vote_average','vote_count','est_rate']]
        if num_recommends > len(recommendations) :
            return recommendations
        else :
            return recommendations.head(num_recommends)

