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
    # if the user accesses a business page, similar businesses will be loaded to the given business
    if business_id:
        city = lookup_city_business(business_id, CITIES)
        similarity_matrix = similarity_matrix_categories(categories_city(BUSINESSES, city))
        selection = select_businesses_similarity(similarity_matrix, business_id, city, n)
        selection = lookup_businesses(selection, city, BUSINESSES)
        return selection
    # If a user accesses the home page, the recommendations for his personal preferences will be loaded
    if user_id:
        city = lookup_city_user(user_id, CITIES)
        predicted_ratings_categories = predict_ratings(similarity_matrix_categories(categories_city(BUSINESSES, city)), pivot_ratings_city(REVIEWS, city, USERS, BUSINESSES), to_predict_city(user_id, city, BUSINESSES))
        visited = visited_businesses(user_id, city, REVIEWS)
        selection = select_businesses(predicted_ratings_categories, visited, city, n)
        selection = lookup_businesses(selection, city, BUSINESSES)
        return(selection)
    # if the homepage is loaded without an user id, random businesses will be returned
    else:
        city = random.choice(CITIES)
        while len(BUSINESSES[city]) < n:
            city = random.choice(CITIES)
    return random.sample(BUSINESSES[city], n)

def select_businesses_similarity(similarity_matrix, business_id, city, n):
    """
    Returns a selection of n businesses similar to the business id that is given, 
    if there a not enough similar businesses there will be randomly selected businesses included. 
    If there are too many similar businesses the system will randomly 
    choose n amount of similar businesses
    """
    # create selection of similar businesses
    selection = similarity_matrix.index[similarity_matrix[business_id] > 0.1].tolist()
    # remove the business for which the selection is made
    if business_id in selection:
        selection.remove(business_id)
    # make sure there are not too many businesses selected
    while len(selection) > n:
        random.shuffle(selection)
        selection.pop()
    # make sure there are sufficient businesses selected
    while len(selection) < n:
        k = n - len(selection)
        random_choice = random.sample(BUSINESSES[city], k)
        # expand selection if necessary
        for choice in random_choice:
            if choice['business_id'] not in selection:
                selection.append(choice['business_id'])
    return selection

def lookup_city_user(user_id, CITIES):
    """
    lookup the city where an user is registered
    """
    for city in CITIES:
        for user in USERS[city]:
            if user['user_id'] == user_id:
                return city

def lookup_city_business(business_id, CITIES):
    """
    lookup the city where an business is registered
    """
    for city in CITIES:
        for business in BUSINESSES[city]:
            if business['business_id'] == business_id:
                return city

def lookup_businesses(selection, city, BUSINESSES):
    """
    Convert list of business ids to list of dictionaries containing the required business information
    """
    selection_complete = []
    for business in BUSINESSES[city]:
        if business['business_id'] in selection:
            selection_complete.append({'business_id': business['business_id'], 'stars': business['stars'], 'name': business['name'], 'city': city, 'address': business['address']})
    return selection_complete

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
    # select the businesses for which the predicted rating is positive
    for index, row in prediction_matrix.iterrows():
        if row['predicted rating'] > 3.0 and row['business_id'] not in visited:
            selection.append(row['business_id'])
    # make sure selection is not too big
    while len(selection) > n:
        random.shuffle(selection)
        selection.pop()
    # make sure selection is sufficient, if needed expand selection with random input
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
