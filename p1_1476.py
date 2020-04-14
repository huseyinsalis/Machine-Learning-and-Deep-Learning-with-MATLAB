# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:29:30 2020

@author: user1
"""

import pandas as pd
#import numpy as np
#import datetime as dt

# Make display smaller
pd.options.display.max_rows = 10

unames = ['user_id', 'gender', 'age', 'occupation', 'zip'] 
users = pd.read_table('users.dat', sep='::',
                      header=None, names=unames)
rnames = ['user_id', 'movie_id', 'rating', 'timestamp'] 
ratings = pd.read_table('ratings.dat', sep='::',                   
                        header=None, names=rnames)
mnames = ['movie_id', 'title', 'genres'] 
movies = pd.read_table('movies.dat', sep='::',                       
                       header=None, names=mnames) 

"""Problem 1:
    Show total number of rows with missing values (after join all files), 
and remove those rows from the DataFrame"""

data = pd.merge(pd.merge(ratings, users, how='outer'), movies, how='outer') #Merge all tables with missing values
data_counts = data[::].shape[0] #Count the combined table
data2 = data.dropna() #Drop all missing value rows
data_counts2 = data2[::].shape[0] #Count the combined table without missing values

print("Total number of rows with missing values: " , data_counts)
print("Total number of rows after dropping missing values: " , data_counts2)

#------------------------------------------------
"""Problem 2: Show total number of movies, total number of genres, 
    total number of users, total number of ratings."""
genres_unique = pd.DataFrame(movies.genres.str.split('|').tolist()).stack().unique()
genres_unique = pd.DataFrame(genres_unique, columns=['genre'])
   
print("Total number of movies: ", data['movie_id'].nunique())
print("Total number of genres: ", genres_unique['genre'].nunique())
print("Total number of users: ", data['user_id'].nunique())
print("Total number of ratings: ", ratings['rating'].count())

#------------------------------------------------
"""Problem 3: ) Plot the bar charts or density estimates for number of ratings per user, 
number of movies per year,number of movies per genre,
average ratings per movie,average ratings per genre """

#By grouping by user id we can find number of ratings
grouped_by_user = data2.filter(['user_id','rating']).groupby(['user_id']).count() 
grouped_by_user.plot.bar(rot=0)

#------------------------------------------------
# Number of Movies per year
movies['year'] = movies['title'].str.extract(r'[(](\d\d\d\d)' ).astype(int)
grouped_by_year = movies.filter(['movie_id', 'year']).groupby(['year']).count() 
grouped_by_year.plot.bar(rot=0)

#------------------------------------------------
#Finding number of movies per genre
movies = movies.join(movies.genres.str.get_dummies().astype(bool))
#movies.drop('genres', inplace=True, axis=1)
numberofmoviespergenre = pd.DataFrame(index=genres_unique['genre'])
totalmovies=[]
for genre in genres_unique.genre:
    totalmovies.append(movies[genre].sum())
numberofmoviespergenre['NumberofMovies'] = totalmovies
numberofmoviespergenre.plot.bar(rot=90)

#------------------------------------------------
#Average Ratings per Movie
avg_rating_per_movie = data2.filter(['movie_id', 'rating']).groupby('movie_id').agg(['mean'])
avg_rating_per_movie.plot.kde()


#------------------------------------------------
#Average Ratings per Genre
movies_with_ratings = pd.merge(movies, ratings, how='left')
Averageratingpergenre = pd.DataFrame(index=genres_unique['genre'])
AverageforGenres=[]
for genre in genres_unique.genre:
    genre_places = movies_with_ratings[genre] == True
    AverageforGenres.append(movies_with_ratings.loc[genre_places, 'rating'].mean())
Averageratingpergenre['Average_Rating'] = AverageforGenres
Averageratingpergenre.plot.bar(rot=90)

#------------------------------------------------
"""Problem 4.1 :Show the number of users whose number of ratings are greater or 
    equal to the median of the number of ratings"""
# first we need to find number of ratings per user
median_of_ratings = grouped_by_user['rating'].median()
result= grouped_by_user[grouped_by_user['rating'] >= median_of_ratings]
print("Number of users whose number of ratings are greater or equal to the median of the number of ratings: \n", result.shape[0])

#------------------------------------------------
"""Problem 4.2 : Show the top ten movies with title and genres 
rated by each user from (1)
"""
result['user_id']=result.index.astype(int)
user_list=result['user_id'].tolist()
avg_rating_per_movie['movie_id']=avg_rating_per_movie.index
avg_rating_per_movie=avg_rating_per_movie.reset_index(drop=True)
avg_rating_per_movie.columns = avg_rating_per_movie.columns.get_level_values(0)
avg_rating_per_movie.columns = ['avg_rating', 'movie_id']
result1 = pd.merge(data2, avg_rating_per_movie, how='left')
result1['user_id'] = result1['user_id'].astype(int)

result2=result1[result1['user_id'].isin(user_list)]
result3 = result2.sort_values(by=['avg_rating'], ascending=False)
result3.drop_duplicates(subset ="movie_id", keep = 'first', inplace = True)
result4 = result3[['title', 'genres', 'avg_rating']].head(10)
print("top ten movies rated by each user from (1) :", result4.to_string(index=False))

#------------------------------------------------
"""Problem 4.3 :Show the average rating per genra for each user from (1)
"""
data3=data2[data2['user_id'].astype(int).isin(result['user_id'])]
data4=pd.merge(data3,movies_with_ratings,how='left')
data4['user_id']=data4['user_id'].astype(int)
useridc=data4['user_id'].unique()
columnlist=['user_id']+list(genres_unique.set_index('genre').T)
AverageforGenresforuser=pd.DataFrame(columns=columnlist)

for i in useridc:
    data5=data4[data4['user_id']==i]
    avrg=[i]
    for genre in genres_unique.genre:
        if any(data5[genre]):
            genre_places = data5[genre] == True
            avrg.append(data5.loc[genre_places, 'rating'].mean())
        else:
            avrg.append(0)
    AverageforGenresforuser.loc[len(AverageforGenresforuser)]=avrg
print("Average Rating per Genre for each user from (1)\n ", AverageforGenresforuser)