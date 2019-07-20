import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LsiModel
from gensim.models import TfidfModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.similarities import MatrixSimilarity
from gensim.matutils import corpus2dense
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise.evaluate import evaluate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

class ContentBasedRecommenders :

    def __init__(self):
        pd.options.mode.chained_assignment = None

        ratings_path = './data/ratings_small.csv'
        md_path = './data/movies_metadata.csv'
        links_path = './data/links_small.csv'

        md = pd.read_csv(md_path,low_memory=False,
                         usecols=['genres','id','overview','production_companies','status','tagline','title','vote_average','vote_count'],
                         dtype={'tagline':str,'overview':str},na_values={'vote_average':0,'vote_count':0,'status':['Canceled','In Production','Planned','Rumored']})
        md.drop(labels=[19730, 29503, 35587], axis=0, inplace=True)
        md.dropna(subset=['status'], axis=0, inplace=True)
        md['id'] = md['id'].astype(int)

        links = pd.read_csv(links_path)
        links.dropna(axis=0, subset=['tmdbId'], inplace=True)

        ids = links['tmdbId'].astype(int)
        self.smd = md[md['id'].isin(ids)]
        self.smd.reset_index(drop=True, inplace=True)

        self.ratings = pd.read_csv(ratings_path)

        reader = Reader(rating_scale=(1, 5))

        dataset = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], reader)

        dataset.split(n_folds=5)

        self.svd = SVD()


        # evaluate(svd,dataset,measures=['rmse','mae'])

        trainset = dataset.build_full_trainset()

        self.svd.fit(trainset)
        self.__tagline_set_up()
        self.__info_set_up()

        print("Recommenders ready.")

    def __extract_names(self,name, doc):
        #Used to extract names

        for sent in nltk.sent_tokenize(doc):
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(doc))):
                try:
                    if chunk.label() == 'PERSON':
                        for c in chunk.leaves():
                            if str(c[0]).lower() not in name:
                                name.append(str(c[0]))

                except AttributeError:
                    pass
        return name


    def __process_data(self,descriptions):
        # tokenize, remove stop words, stem
        # remove non-letters and characters
        descriptions = descriptions.apply(lambda x: re.sub("[^a-zA-Z]", " ", x))
        stopWords = set(stopwords.words('english'))
        stopWords.update(['film', 'films', 'story', 'stories', 'man', 'woman', 'boy', 'girl'])
        #For speed sake using predefined list of names created using extract_names function
        names_df = pd.read_csv('data/names_list.csv', usecols=['0'])
        names = list(names_df['0'])

        # for description in descriptions:
        #    names = self.__extract_names(names, description)

        stopWords.update(names)

        stemmer = SnowballStemmer('english')

        descriptions = descriptions.apply(lambda x: x.split())

        for i, descrip in descriptions.iteritems():
            descriptions.loc[i] = [stemmer.stem(word.lower()) for word in descrip if word not in stopWords]

        return descriptions

    def __prepare_corpus(self,clean_doc):

        dictionary = corpora.Dictionary(clean_doc)

        doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_doc]

        tfidf = TfidfModel(doc_term_matrix)
        corpus_tfidf = tfidf[doc_term_matrix]

        return dictionary, corpus_tfidf

    def __create_lsa_model(self,clean_doc, num_topics, words):

        dictionary, corpus_tfidf = self.__prepare_corpus(clean_doc)

        # generate LSA model
        lsamodel = LsiModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary)

        print("Topics: ", lsamodel.print_topics(num_topics=num_topics, num_words=words))

        return lsamodel

    def __tagline_set_up(self):
        #Preparing data for description based recommender

        print("Cleaning tagline data...")

        self.smd['tagline'] = self.smd['tagline'].fillna('')
        self.smd['overview'] = self.smd['overview'].fillna('')
        self.smd['description'] = self.smd['overview'] + self.smd['tagline']

        self.smd['description'] = self.__process_data(self.smd['description'])

        #Determining ideal number of topics
        # start,stop,step = 1,45,5
        # plot_coher_vals(small_md['description'],start,stop,step)
        # dictionary,doc_term_matrix = prepare_corpus(small_md['description'])

        # num of optimal topics=19

        print("Training LSI model...")
        lsa_model = self.__create_lsa_model(self.smd['description'], 30, 4)
        dictionary, corpus_tfidf = self.__prepare_corpus(self.smd['description'])

        print("Calculating tagline similarities...")
        self.smd['similarity'] = 'unknown'
        self.smd['size_similar'] = 0
        self.smd = self.smd.astype(object)
        threshold = 0.25
        all_sims = []
        index = MatrixSimilarity(lsa_model[corpus_tfidf])
        for j, movie in enumerate(corpus_tfidf):
            lsi_vec = lsa_model[movie]
            sim = index[lsi_vec]
            all_sims = np.concatenate([all_sims, sim])
            sim_list = []
            for i, id in enumerate(self.smd['id']):
                if (sim[i] > threshold):
                    sim_list.append((id, sim[i]))
            sim_list = sorted(sim_list, key=lambda item: -item[1])
            self.smd.at[j, 'similarity'] = sim_list
            self.smd.at[j, 'size_similar'] = len(sim_list)

    def __info_set_up(self):
        #Preparing data for metadata recommender

        credits_path = './data/credits.csv'
        keywords_path = './data/keywords.csv'

        print("Creating metadata recommender...")

        creds = pd.read_csv(credits_path, dtype={'id': int})
        keys = pd.read_csv(keywords_path, dtype={'id': int})

        creds['id'] = creds['id'].astype(int)
        keys['id'] = keys['id'].astype(int)

        self.smd = self.smd.merge(creds, on='id')
        self.smd = self.smd.merge(keys, on='id')

        # get top 3 actors and director
        self.smd['cast'] = self.smd['cast'].apply(
            lambda c_row: list(map(lambda c_item: c_item['name'].strip().lower(), ast.literal_eval(c_row)))[:3])
        self.smd['director'] = self.smd['crew'].apply(
            lambda c_row: [c_item['name'].strip().lower() for c_item in ast.literal_eval(c_row) if
                           c_item['job'] == 'Director'])
        self.smd['director_dup'] = self.smd['director'].apply(lambda d_list: d_list + d_list + d_list)

        # get keywords
        self.smd['keywords'] = self.smd['keywords'].apply(
            lambda k_row: list(map(lambda k_item: k_item['name'].strip().lower(), ast.literal_eval(k_row))))
        stemmer = SnowballStemmer('english')
        self.smd['keywords'] = self.smd['keywords'].apply(lambda k_list: [stemmer.stem(word) for word in k_list])

        # get genres
        self.smd['genres'] = self.smd['genres'].apply(
            lambda g_row: list(map(lambda g_item: g_item['name'].strip().lower(), ast.literal_eval(g_row))))

        self.smd['mix'] = self.smd['cast'] + self.smd['director'] + self.smd['keywords'] + self.smd['genres']
        self.smd['mix'] = self.smd['mix'].apply(lambda m_list: " ".join(m_list))

        cv = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.95, min_df=0.05)
        doc_term_matrix = cv.fit_transform(self.smd['mix'])
        self.cos_similarity = cosine_similarity(doc_term_matrix, doc_term_matrix)


    def __weighted_rating(self,movie, C, m):
        v = movie['vote_count']
        R = movie['vote_average']
        return (R * (v / (v + m)) + C * (m / (v + m)))

    def __convert_int(self,number):
        try:
            return int(number)
        except:
            return np.nan

    def recommend_tagline(self,movie_title,genre=None):
        #Input: movie title, genre(optional)
        #Output: 25 recommendations from topic model trained on movie overviews and taglines
        metadata = self.smd.set_index('id')
        movie_metadata = metadata[metadata['title'] == movie_title].reset_index()
        if movie_metadata.empty :
            print("Movie: ",movie_title," not in data set. Please select another movie.")
            return

        similarities = movie_metadata['similarity'][0][1:]
        sim_ids = [movie[0] for movie in similarities]
        similar_movies = metadata.loc[sim_ids]
        similar_movies.dropna(subset=['vote_average', 'vote_count'], axis=0, inplace=True)
        if genre :
            genres_series = similar_movies['genres']
            genre_movies = [label for label, movie in similar_movies.iterrows() if
                            genre.lower() in genres_series[label]]
            similar_movies = similar_movies.loc[genre_movies, :]
            if similar_movies.empty :
                print("Invalid genre")
                return
        C = similar_movies['vote_average'].mean()
        m = similar_movies['vote_count'].astype(int).quantile(q=0.4)
        qualified = similar_movies[similar_movies['vote_count'] >= m][0:25]
        qualified['weighted_rating'] = 0.0
        for j, movie in qualified.iterrows():
            wr = self.__weighted_rating(movie, C, m)
            qualified.at[j, 'weighted_rating'] = wr
        recommendations = qualified.sort_values(by='weighted_rating', ascending=False)
        return recommendations.loc[:,['title','overview','genres','cast','director','weighted_rating','vote_average','vote_count']]

    def recommend_info(self,movie_title,genre=None):
        #Input: movie title, genre (optional)
        #Output: Returns recommendations for similar movies by determining cosine similarity between movie metadata (director, genres, keywords, cast and crew)

        metadata = self.smd
        movie_metadata = metadata[metadata['title'] == movie_title]
        if movie_metadata.empty :
            print("Movie: ",movie_title," not in data set. Please select another movie.")
            return
        similarities = np.array(self.cos_similarity[movie_metadata.index, :])
        similarities_sorted = list(np.argsort(-similarities).flatten())
        similar_movies = metadata.iloc[similarities_sorted, :].dropna(subset=['vote_average', 'vote_count'], axis=0)
        similar_movies.drop(index=similar_movies[similar_movies['title'] == movie_title].index, inplace=True)
        if genre :
            genres_series = similar_movies['genres']
            genre_movies = [label for label, movie in similar_movies.iterrows() if
                            genre.lower() in genres_series[label]]
            similar_movies = similar_movies.loc[genre_movies, :]
            if similar_movies.empty:
                print("Invalid genre")
                return
        C = similar_movies['vote_average'].mean()
        m = similar_movies['vote_count'].astype(int).quantile(q=0.4)
        qualified = similar_movies[similar_movies['vote_count'] >= m][0:25]
        qualified['weighted_rating'] = 0.0
        for j, movie in qualified.iterrows():
            wr = self.__weighted_rating(movie, C, m)
            qualified.at[j, 'weighted_rating'] = wr
        recommendations = qualified.sort_values(by='weighted_rating', ascending=False)
        return recommendations.loc[:,['title','overview','genres','cast','director','weighted_rating','vote_average','vote_count']]

    def recommend_hybrid(self,user_id,movie_title,genre=None):
        #Input: user id, movie title, genre (optional)
        #Output: Gives recommendations for a user based on movie provided, using svd algorithm

        metadata = self.smd
        movie_metadata = metadata[metadata['title'] == movie_title].reset_index(drop=True)
        if movie_metadata.empty :
            print("Movie: ",movie_title," not in data set. Please select another movie.")
            return

        similarities = movie_metadata['similarity'][0][1:]
        sim_ids = [movie[0] for movie in similarities]
        similar_movies = metadata[metadata['id'].isin(sim_ids)].reset_index(drop=True)
        similar_movies['id'] = similar_movies['id'].apply(lambda x: self.__convert_int(x))
        similar_movies.dropna(subset=['id'], inplace=True)

        # Filter out movies the user has already rated/seen
        rated_movie_ids = self.ratings[self.ratings['userId'] == user_id]['movieId']
        similar_movies = similar_movies[~similar_movies['id'].isin(rated_movie_ids)]
        if genre:
            genres_series = similar_movies['genres']
            genre_movies = [label for label, movie in similar_movies.iterrows() if
                            genre.lower() in genres_series[label]]
            similar_movies = similar_movies.loc[genre_movies, :]
            if similar_movies.empty:
                print("Invalid genre")
                return
        similar_movies = similar_movies[0:25]
        similar_movies['est_rate'] = 0.0
        predictions = [self.svd.predict(user_id, m_id) for m_id in similar_movies['id'].values]
        similar_movies['est_rate'] = [est for uid, iid, r_true, est, _ in predictions]
        recommendations = similar_movies.sort_values(by='est_rate', ascending=False)
        return recommendations.loc[:,['title','overview','genres','cast','director','vote_average','vote_count']]