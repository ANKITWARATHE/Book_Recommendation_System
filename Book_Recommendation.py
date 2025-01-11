import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

books = pd.read_csv('books.csv')
users = pd.read_csv('users.csv')
ratings = pd.read_csv('ratings.csv')

books.head()
users.head()
ratings.head()

books.shape
users.shape
ratings.shape


books.isnull().sum()
users.isnull().sum()
ratings.isnull().sum()

books.duplicated().sum()
users.duplicated().sum()
ratings.duplicated().sum()


# Popularity Based Recommender System
ratings_with_name = ratings.merge(books, on = 'ISBN')

## Number of rating per book
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns = {'Book-Rating':'num_ratings'},inplace=True)
num_rating_df

## Average rating per book
avg_rating_df = ratings_with_name.groupby('Book-Title').agg({'Book-Rating': 'mean'}).reset_index()
avg_rating_df.rename(columns = {'Book-Rating':'avg_rating'},inplace=True)
avg_rating_df

popular_df = num_rating_df.merge(avg_rating_df, on = 'Book-Title')

## Top-50 Highest rated books those are getting more then 250 votes -> Popular Books
popular_df = popular_df[popular_df['num_ratings']>=250].sort_values('avg_rating', ascending = False).head(50)

popular_df = popular_df.merge(books, on = 'Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_rating']]

## Top-50 Popular Books
popular_df

# **************************************************************************
## Collaborative Filtering Based Recommender System
# users those are rated More then 200 books
x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
padhe_likhe_users = x[x].index

filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]

# More then 50 rated books
y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = y[y].index

final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

final_ratings

pt = final_ratings.pivot_table(index='Book-Title', columns = 'User-ID', values = 'Book-Rating')
pt.fillna(0, inplace = True)

from sklearn.metrics.pairwise import cosine_similarity
similarity_scores = cosine_similarity(pt)
similarity_scores.shape

# pt_index_df = pd.DataFrame(final_rating.index, index = pt.index)
# pt_index_df = pt_index_df.reset_index()
# pt_index_df.rename({'index':'book_name', 0 :'indexs'}, inplace = True)


def recommend(book_name):
    # index fetch
    index = np.where(pt.index == book_name)[0][0]
   # index = pt_index_df[pt_index_df.book_name = book_name]['indexs']
    similar_items = sorted(list(enumerate(similarity_scores[index])), key = lambda x:x[1], reverse = True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    
    return data


recommend('1984')

import pickle
pickle.dump(popular_df,open('popular.pkl','wb'))

books.drop_duplicates('Book-Title')

pickle.dump(pt,open('pt.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_scores,open('similarity_scores.pkl','wb'))














