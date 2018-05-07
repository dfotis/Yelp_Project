#test
import pymongo

def find_restaurants_by_neighborhood(neighborhood):
    return db['restaurants'].find({'neighborhood': neighborhood})

def find_reviews_that_contain_a_word(word):
    return db['reviews'].find({'text': {'$regex': '.*' + word + '.*'}})

client = pymongo.MongoClient('localhost', 27017)
db = client['yelpdb']

#eastside_restaurants = find_restaurants_by_neighborhood("Eastside")
#for restaurant in eastside_restaurants:
#    print(restaurant)


#match_reviews = find_reviews_that_contain_a_word("awesome")
#for reviews in match_reviews:
#    print(reviews)