from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

import random
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from IPython.display import display
import math

def recommend(user_id=None, business_id=None, city=None, n=10):
    """
    Returns n recommendations as a list of dicts.
    Optionally takes in a user_id, business_id and/or city.
    A recommendation is a dictionary in the form of:
        {
            business_id:str
            stars:str
            name:str
            city:str
            adress:str
        }
    """
    if user_id and city:
        predicted_ratings_categories = predict_ratings(similarity_matrix_categories(categories_city(BUSINESSES, city)), pivot_ratings_city(REVIEWS, city, USERS, BUSINESSES), to_predict_city(user_id, city, BUSINESSES))
        visited = visited_businesses(user_id, city, REVIEWS)
        selection = select_businesses(predicted_ratings_categories, visited, city, n)
        return(selection)
    if not city:
        city = random.choice(CITIES)
    return random.sample(BUSINESSES[city], n)

def visited_businesses(user_id, city, REVIEWS):
    """
    Creates a list of the businesses within the given city for the user
    """
    visited = []
    for review in REVIEWS[city]:
        if review['user_id'] == user_id:
            visited.append(review['business_id'])
    return visited

def select_businesses(prediction_matrix, visited, city, n):
    """
    Returns an selection of n amount of businesses based on the predictions. If there are 
    insufficient positive predictions, the list will be expanded with random business choices. 
    """
    selection = []
    for index, row in prediction_matrix.iterrows():
        if row['predicted rating'] > 3.0 and row['business_id'] not in visited:
            print(row['business_id'], row['predicted rating'])
            selection.append(row['business_id'])
    while len(selection) > n:
        random.shuffle(selection)
        selection.pop()
    while len(selection) < n:
        k = n - len(selection)
        random_choice = random.sample(BUSINESSES[city], k)
        for choice in random_choice:
            if choice['business_id'] not in visited and choice['business_id'] not in selection:
                selection.append(choice['business_id'])
    return selection

def get_rating(REVIEWS, user_id, business_id):
    """
    Returns the rating from a user on a business, if there is one.
    """
    for city in CITIES:
        for review in range(len(REVIEWS[city])):
            reviewdict = REVIEWS[city][review]
            if reviewdict['user_id'] == user_id and reviewdict['business_id'] == business_id:
                rating = (REVIEWS[city][review]['stars'])
                return rating
    return np.nan

def pivot_ratings_city(REVIEWS, city, USERS, BUSINESSES):
    """
    Returns an matrix with the businesses within the city as index and the users as column. 
    The values are the ratings the users gave for each business.
    """
    users = []
    businesses = []
    for user in USERS[city]:
        users.append(user['user_id'])
    for business in BUSINESSES[city]:
        businesses.append(business['business_id'])
    pivot_data = pd.DataFrame(np.nan, columns=users, index=businesses, dtype=float)
    for x in pivot_data:
        for y in pivot_data.index:
            pivot_data.loc[y][x] = get_rating(REVIEWS, x, y)
    return pivot_data

def categories_city(BUSINESSES, city):
    """
    Returns a dataframe with all the businesses and categories and fills in an 1 if the
    categorie is applicable for the business and a 0 if not
    """
    business_categories = pd.DataFrame()
    for business in BUSINESSES[city]:
        try:
            for categorie in business['categories'].split(','):
                business_categories.loc[business['business_id'], categorie] = 1
        except:
            pass
    business_categories = business_categories.fillna(0)
    return business_categories

def similarity_matrix_categories(matrix):
    """
    creates a similarity matrix based on the categories of businesses
    """
    npu = matrix.values
    m1 = npu @ npu.T
    diag = np.diag(m1)
    m2 = m1 / diag
    m3 = np.minimum(m2, m2.T)
    return pd.DataFrame(m3, index = matrix.index, columns = matrix.index)

def predict_ratings(similarity, utility, to_predict):
    """
    Predict the ratings for each user and business combination in the to predict matrix.
    """
    # copy input (don't overwrite)
    ratings_test_c = to_predict.copy()
    # apply prediction to each row
    ratings_test_c['predicted rating'] = to_predict.apply(lambda row: predict_ids(similarity, utility, row['user_id'], row['business_id']), axis=1)
    return ratings_test_c

def to_predict_city(user_id, city, BUSINESSES):
    """
    Creates a matrix containing the user id in one column and all the businesses in a city in 
    the other column.
    """
    businesses = []
    for business in BUSINESSES[city]:
        businesses.append(business['business_id'])
    to_predict_df = pd.DataFrame(columns=['user_id', 'business_id'])
    to_predict_df.loc[:, 'business_id'] = businesses
    to_predict_df = to_predict_df.fillna(user_id)
    return to_predict_df

def predict_ids(similarity, utility, userId, itemId):
    # select right series from matrices and compute
    if userId in utility.columns and itemId in similarity.index:
        return predict_vectors(utility.loc[:,userId], similarity[itemId])
    return 0

def predict_vectors(user_ratings, similarities):
    # select only movies actually rated by user
    relevant_ratings = user_ratings.dropna()
    
    # select corresponding similairties
    similarities_s = similarities[relevant_ratings.index]
    
    # select neighborhood
    similarities_s = similarities_s[similarities_s > 0.0]
    relevant_ratings = relevant_ratings[similarities_s.index]
    
    # if there's nothing left return a prediction of 0
    norm = similarities_s.sum()
    if(norm == 0):
        return 0
    
    # compute a weighted average (i.e. neighborhood is all) 
    return np.dot(relevant_ratings, similarities_s)/norm

predicted_ratings_categories = predict_ratings(similarity_matrix_categories(categories_city(BUSINESSES, 'ambridge')), pivot_ratings_city(REVIEWS, 'ambridge', USERS, BUSINESSES), to_predict_city('v2hPERrqGGxsfmT293q4mA', 'ambridge', BUSINESSES))
display(predicted_ratings_categories)
recommendations = recommend(user_id = 'v2hPERrqGGxsfmT293q4mA', city = 'ambridge', n = 10)
print(recommendations)

def categories(BUSINESSES, CITIES):
    business_categories = pd.DataFrame()
    for city in CITIES:
        for business in BUSINESSES[city]:
            try:
                for categorie in business['categories'].split(','):
                    business_categories.loc[business['business_id'], categorie] = 1
            except:
                pass
    business_categories = business_categories.fillna(0)
    return business_categories

def to_predict(user_id, CITIES, BUSINESSES):
    businesses = []
    for city in CITIES:
        for business in BUSINESSES[city]:
            businesses.append(business['business_id'])
    to_predict_df = pd.DataFrame(columns=['user_id', 'business_id'])
    to_predict_df.loc[:, 'business_id'] = businesses
    to_predict_df = to_predict_df.fillna(user_id)
    return to_predict_df

def pivot_ratings(REVIEWS, CITIES, USERS, BUSINESSES):
    users = []
    businesses = []
    for city in CITIES:
        for user in USERS[city]:
            users.append(user['user_id'])
        for business in BUSINESSES[city]:
            businesses.append(business['business_id'])
    pivot_data = pd.DataFrame(np.nan, columns=users, index=businesses, dtype=float)
    for x in pivot_data:
        for y in pivot_data.index:
            pivot_data.loc[y][x] = get_rating(REVIEWS, x, y)
    return pivot_data




def cosine_distance(matrix, id1, id2):
    """Compute cosine distance between two rows."""    
    # only take the features that have values for both id1 and id2
    selected_features = matrix.loc[id1].notna() & matrix.loc[id2].notna()
    
    # if no matching features, return NaN
    if not selected_features.any():
        return np.nan
    
    # get the features from the matrix
    features1 = matrix.loc[id1][selected_features]
    features2 = matrix.loc[id2][selected_features]
    
    # compute the distances for the features
    distances = Series()
    sqrt1 = 0
    sqrt2 = 0
    for feature in features1.index:
        if feature in features2.index or features1[feature]!=0.0 or features1[feature]!=0.0:
            distances.loc[feature] = features1[feature] * features2[feature]
            sqrt1 += features1[feature]**2
            sqrt2 += features2[feature]**2
    distsum = distances.sum()
    if distsum == 0.0:
        return 0.0
    sqrtval = math.sqrt(sqrt1) * math.sqrt(sqrt2)
    #     print(distsum, sqrt1, sqrt2, sqrtval)
    total = distsum / sqrtval
    return total

def cosine_similarity(matrix, id1, id2):
    """Compute cosine similarity between two rows."""
    # compute distance
    distance = cosine_distance(matrix, id1, id2)
    
    # if no distance could be computed (no shared features) return a similarity of 0
    if distance is np.nan:
        return 0
    
    # else return similarity
    return  distance

def create_similarity_matrix_cosine(matrix):
    """ creates the similarity matrix based on cosine similarity """
    similarity_matrix = pd.DataFrame(0, index=matrix.index, columns=matrix.index, dtype=float)
    # select and fill every cell of matrix
    for movieId_ver in similarity_matrix:
        for movieId_hor in similarity_matrix.index:
            if movieId_ver == movieId_hor:
                similarity_matrix.loc[movieId_hor][movieId_ver] = 1.0
            else:
                similarity_matrix.loc[movieId_hor][movieId_ver] = cosine_similarity(matrix, movieId_ver, movieId_hor)
    return similarity_matrix

# df_similarity_ratings = create_similarity_matrix_cosine(pivot_ratings(REVIEWS, CITIES, USERS, BUSINESSES))
# display(df_similarity_ratings)








def pivot_ratings_friends(user_id, REVIEWS, CITIES, USERS, BUSINESSES):
    """
    Return matrix containing all ratings of friends on businesses they have been to
    """
    users = find_friends(user_id, USERS)
    users.append(user_id)
    businesses = []
    for friend in users:
        friends_businesses = check_businesses(friend, REVIEWS)
        for business in friends_businesses:
            businesses.append(business)
    businesses = list(set(businesses))
    pivot_data = pd.DataFrame(np.nan, columns=users, index=businesses, dtype=float)
    for x in pivot_data:
        for y in pivot_data.index:
            pivot_data.loc[y][x] = get_rating(REVIEWS, x, y)
    return pivot_data

def find_friends(user_id, USERS):
    """
    return list of friends for a given user id
    """
    for city, users in USERS.items():
        for user in users:
            if user["user_id"] == user_id:
                friends = user["friends"].split()
    return friends

def check_businesses(user_id, REVIEWS):
    """
    returns a list of businesses a user has placed reviews for
    """
    businesses = []
    for city, reviews in REVIEWS.items():
        for review in reviews:
            if review["user_id"] == user_id:
                businesses.append(review['business_id'])
    return businesses

# rate = get_rating(REVIEWS, 'DAIpUGIsY71noX0wNuc27w', 'PNzir9TtJAD7U41GwR98-w')
# print(REVIEWS['sun city'][0])
# print(f'The rating is {rate}')

# friends = find_friends('MM4RJAeH6yuaN8oZDSt0RA', USERS)
# businesses = check_businesses('LisTsUqnQ5RoW6reg6hyWQ', REVIEWS)
# print(businesses)

# utility_matrix = pivot_ratings_friends('rCWrxuRC8_pfagpchtHp6A', REVIEWS, CITIES, USERS, BUSINESSES)
# display(utility_matrix)

# utility_matrix = pivot_ratings_city('ambridge', REVIEWS, CITIES, USERS, BUSINESSES)
# display(utility_matrix)