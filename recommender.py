from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

import random
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from IPython.display import display

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


def pivot_ratings(ratings, city):
    users = []
    businesses = []
    for user in USERS[city]:
        users.append(user['user_id'])
    for business in BUSINESSES[city]:
        businesses.append(business['business_id'])
    pivot_data = pd.DataFrame(np.nan, columns=users, index=businesses, dtype=float)
    for x in pivot_data:
        for y in pivot_data.index:
            pivot_data.loc[y][x] = get_rating(ratings, x, y)
    return pivot_data

rate = get_rating(REVIEWS, 'DAIpUGIsY71noX0wNuc27w', 'PNzir9TtJAD7U41GwR98-w')
# print(REVIEWS['sun city'][0])
# print(f'The rating is {rate}')
utility_matrix = pivot_ratings(REVIEWS, 'sun city')
display(utility_matrix)