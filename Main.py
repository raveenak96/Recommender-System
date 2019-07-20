from SimpleRecommender import SimpleRecommender
from CollaborativeFiltering import CollaborativeFilter
from ContentBasedRecommenders import ContentBasedRecommenders

# *******************************SIMPLE RECOMMENDER**************************************************
#Gives recommendations by simply returning the most popular movies in the dataset, using IMDB weighted rating formula

simple_recommender = SimpleRecommender()
#Get the top chart
top_chart = simple_recommender.build_chart(20)
#Get 10 general recommendations
s_recommendations = simple_recommender.recommend(10)
#Get 10 romance recommendations
s_rom_recommendations = simple_recommender.recommend(10,'romance')

# ******************************COLLABORATIVE FILTERING***********************************************
#Gives recommendations for users in the dataset

collaborative_filter = CollaborativeFilter()

#Get 20 general recommendations for user 10
user10_recommends = collaborative_filter.recommend(10,20)
#Narrow to romances
user10_recommends_rom = collaborative_filter.recommend(10,'romance',20)

#Get 20 general recommendations for user 1
user1_recommends = collaborative_filter.recommend(1,num_recommends=20)

# *******************************CONTENT BASED/COLLABORATIVE FILTERING RECOMMENDERS********************
#Three recommendation systems:
#1. Movie tagline and overview content based recommender: recommend_tagline
#2. Movie metadata (genres, cast/crew, keywords) content based recommender: recommend_info
#3. Hybrid content based & collaborative filtering recommender: recommend_hybrid

recommender = ContentBasedRecommenders()

#Get recommendations for movies similar to Clueless, based on movie taglines and overviews
t_recommendations = recommender.recommend_tagline('Clueless')
#Narrow The Dark Knight to thrillers
t_thriller_recommendations = recommender.recommend_tagline('The Dark Knight','thriller')

#Get recommendations for movies similar to The Dark Knight, based on movie metadata
i_recommendations = recommender.recommend_info('The Dark Knight')
#Narrow to comedies
i_com_recommendations = recommender.recommend_info('The Dark Knight','comedy')


#Get recommendations for movies similar to The Dark Knight for user number 12 in our dataset
h_recommendations = recommender.recommend_hybrid(12,'The Dark Knight')
#Narrow to crime movies
h_crime_recommendations = recommender.recommend_hybrid(12,'The Dark Knight','crime')

#We can see that the metadata based recommender provides the most accurate recommendations