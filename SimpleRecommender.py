import pandas as pd
import ast

# Simple Recommender that returns the most popular movies in the dataset every time, using full MovieLens Dataset

class SimpleRecommender :

    def __init__(self) :
        metadata_path = './data/movies_metadata.csv'

        #not recommending movies that aren't at least in post-production
        raw_metadata = pd.read_csv(metadata_path,low_memory=False,usecols=['id','overview','genres','imdb_id','popularity','revenue','status','title','vote_average','vote_count'],na_values={'vote_average':0,'vote_count':0,'status':['Canceled','In Production','Planned','Rumored']})

        clean_metadata = raw_metadata
        clean_metadata.dropna(inplace=True, subset=['vote_average', 'vote_count', 'status'])
        clean_metadata['genres'] = clean_metadata['genres'].fillna(value='[]', axis=0)
        clean_metadata['genres'] = clean_metadata['genres'].apply(lambda g_row: list(map(lambda g_item: g_item['name'].lower(), ast.literal_eval(g_row))))

        # Using IMDB Weighted Average Ratings Formula:
        # weighted_rating = R(v/(v+m)) + C(m/(v+m))
        # R = average rating for the movie
        # v = number of votes for the movie
        # m = min number of votes required to be listed in the top chart
        # C = the mean vote across the whole report
        # Will use m=90th percentile (movie has to have more votes than 90% of movies in the report)

        self.C = clean_metadata['vote_average'].mean()
        self.m = clean_metadata['vote_count'].astype(int).quantile(q=0.9)

        self.in_chart = clean_metadata[clean_metadata['vote_count']>=self.m]
        self.top_chart = pd.DataFrame()

    def __weighted_rating(self,movie):
        #helper function to calculate weighted rating

        v = movie['vote_count']
        R = movie['vote_average']
        return (R * (v / (v + self.m)) + self.C * (self.m / (v + self.m)))

    #build a chart of the top num_items movies (general or genre-specific)
    def build_chart(self,num_items,genre=None,return_chart=True):
        #Input: number of items, genre, whether to return the chart
        #Output: returns a chart of the most popular movies in our dataset

        self.top_chart = self.in_chart
        if genre:
            for i,row in self.top_chart.iterrows() :
                if genre.lower() not in row['genres'] :
                    self.top_chart.drop(labels=i,axis=0,inplace=True)
        self.top_chart['weighted_rating'] = self.top_chart.apply(lambda movie: self.__weighted_rating(movie), axis=1)
        self.top_chart = self.top_chart.sort_values(by='weighted_rating', ascending=False).head(num_items)
        if return_chart :
            return self.top_chart

    #return general or genre-specific recommendations
    def recommend(self,num_recommends=25,genre=None) :
        #Input: num recommendations, genre(optional)
        #Returns num_recommends recommendations based on most popular movies

        if self.top_chart.empty or genre :
            self.top_chart = self.build_chart(250,genre)
        if num_recommends > len(self.top_chart) :
            return self.top_chart
        else :
            return self.top_chart.head(num_recommends).loc[:,['title','overview','genres','weighted_rating','vote_average','vote_count']]

