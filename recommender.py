from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

import random
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from IPython.display import display
import csv

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


    if not city:
        city = random.choice(CITIES)
    return random.sample(BUSINESSES[city], n)

def get_rating(REVIEWS, user_id, business_id):
    for city in CITIES:
        for review in range(len(REVIEWS[city])):
            reviewdict = REVIEWS[city][review]
            if reviewdict['user_id'] == user_id and reviewdict['business_id'] == business_id:
                rating = (REVIEWS[city][review]['stars'])
                return rating
    return np.nan


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

def pivot_ratings_city(city, REVIEWS, CITIES, USERS, BUSINESSES):
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

# utility_matrix = pivot_ratings_friends('QGgWWhEi5R4SLAKN-xwtNQ', REVIEWS, CITIES, USERS, BUSINESSES)
# display(utility_matrix)

# utility_matrix = pivot_ratings_city('sun city', REVIEWS, CITIES, USERS, BUSINESSES)
# display(utility_matrix)