#test
import pymongo
import string
import datetime
import re
import numpy as np
from collections import Counter
from bson.json_util import dumps
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import SnowballStemmer
import pandas as pd

from sklearn.model_selection import train_test_split

#import numpy as np
#import seaborn as sns

import nltk
from nltk.corpus import stopwords

client = pymongo.MongoClient('localhost', 27017)
db = client['yelpdb']

def save_to_mongo(collection_name, custom_object):
    try:
        collection = db[collection_name]
        result = collection.insert_one(custom_object).inserted_id
        # print("Saved successfully.")
    except pymongo.errors.ConnectionFailure as e:
        print("Could not connect to MongoDB: %s" % e)

def find_restaurants_by_neighborhood(neighborhood):
    return db['restaurants'].find({'neighborhood': neighborhood})

def find_reviews_that_contain_a_word(word):
    return db['reviews'].find({'text': {'$regex': '.*' + word + '.*'}})

def find_reviews_for_specific_restaurant(restaurant_id):
    return db['reviews'].find({'business_id': restaurant_id})

def find_restaurants_by_category(category):
    return db['restaurants'].find({"$or":[ {"categories.0":category}, {"categories.1":category}, {"categories.2":category},
                                           {"categories.3":category}, {"categories.4":category}, {"categories.5": category},
                                           {"categories.6":category}, {"categories.7":category}, {"categories.8":category},
                                           {"categories.9":category}, {"1categories.0": category}, {"categories.11":category},
                                           {"categories.12":category}]})



#eastside_restaurants = find_restaurants_by_neighborhood("Eastside")
#for restaurant in eastside_restaurants:
#    print(restaurant)

'''
    Collecting only Italian Restaurants
'''
def select_italian_restaurants():
    for rest in find_restaurants_by_category('Italian'):
        custom_object = {
            "name": rest['name'],
            "business_id": rest['business_id'],
            "longitude": rest['longitude'],
            "latitude": rest['latitude'],
            "neighborhood": rest['neighborhood'],
            "stars": rest['stars'],
            "attributes": rest['attributes'],
            "review_count": rest['review_count']}
        save_to_mongo("Italian_Restaurants", custom_object)

        for review in find_reviews_for_specific_restaurant(rest['business_id']):
            custom_object = {
                "user_id": review['user_id'],
                "review_id": review['review_id'],
                "text": review['text'],
                "business_id": review['business_id'],
                "stars": review['stars'],
                "date": review['date'],
                "useful": review['useful'],
                "funny": review['funny'],
                "cool": review['cool']}
            save_to_mongo("Italian_Reviews", custom_object)

            for user in db['users'].find({'user_id': review['user_id']}):
                custom_object = {
                    "yelping_since": user['yelping_since'],
                    "useful": user['useful'],
                    "compliment_photos": user['compliment_photos'],
                    "compliment_list": user['compliment_list'],
                    "compliment_funny": user['compliment_funny'],
                    "funny": user['funny'],
                    "review_count": user['review_count'],
                    "friends": user['friends'],
                    "fans": user['fans'],
                    "compliment_note": user['compliment_note'],
                    "compliment_plain": user['compliment_plain'],
                    "compliment_writer": user['compliment_writer'],
                    "compliment_cute": user['compliment_cute'],
                    "average_stars": user['average_stars'],
                    "user_id": user['user_id'],
                    "compliment_more": user['compliment_more'],
                    "elite": user['elite'],
                    "compliment_hot": user['compliment_hot'],
                    "name": user['name'],
                    "cool": user['cool'],
                    "compliment_profile": user['compliment_profile'],
                    "compliment_cool": user['compliment_cool']}
                save_to_mongo("Italian_Users", custom_object)



'''
    Load text and stars from Italian restaurants reviews
'''
def add_preprocessed_field_to_reviews():
    count = 0
    for review in db['Italian_Reviews'].find():
        preprocessed_text = text_process(review['text'])
        db['Italian_Reviews'].update({"_id": review["_id"]}, {"$set": {"Preprocessed_Text": preprocessed_text}})

        count += 1
        print(count)



def classifier_MultinomialNB(X_train, X_test, y_train, y_test):
    print("Fitting the classifier...")
    classifier = MultinomialNB()

    classifier.fit(X_train, y_train)

    preds = classifier.predict(X_test)

    print(classification_report(y_test, preds))



def classifier_LinearSVC(X_train, X_test, y_train, y_test):
    print("Fitting the classifier...")
    classifier = LinearSVC()

    # train the classifier
    t1 = datetime.datetime.now()
    classifier.fit(X_train, y_train)
    print(datetime.datetime.now() - t1)

    #print("Predictions...")
    preds = classifier.predict(X_test)

    #print(list(preds[:10]))
    #print(y_test[:10])

    print(classification_report(y_test, preds))

def balance_classes(xs, ys):
    """Undersample xs, ys to balance classes."""
    freqs = Counter(ys)

    # the least common class is the maximum number we want for all classes
    max_allowable = freqs.most_common()[-1][1]
    num_added = {clss: 0 for clss in freqs.keys()}
    new_ys = []
    new_xs = []
    for i, y in enumerate(ys):
        if(num_added[y] < max_allowable):
            new_ys.append(y)
    new_xs.append(xs[i])
    num_added[y] += 1
    return new_xs, new_ys

def initialize_db():
    client = pymongo.MongoClient('localhost', 27017)
    db = client['yelpdb']

    return db

def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    '''
    stemmer = SnowballStemmer("english")

    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    return [stemmer.stem(word.lower()) for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def main():
    db = initialize_db()

    #add_preprocessed_field_to_reviews()
    reviews = list(db['Italian_Reviews'].find())
    yelp_reviews = pd.read_json(dumps(reviews))

    ## Select only reviews with 1 star and 5 stars
    yelp_reviews = yelp_reviews[(yelp_reviews['stars'] == 1) | (yelp_reviews['stars'] == 2) | (yelp_reviews['stars'] == 4) | (yelp_reviews['stars'] == 5)]
    print("Reviews loaded.")



    #reviews = pd.DataFrame(list(db['Italian_Reviews'].find()))



    # This vectorizer breaks text into single words and bi-grams
    # and then calculates the TF-IDF representation
    print("Vectorization started...")

    #vectorizer = TfidfVectorizer(min_df=.0025, max_df=.1, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
    t1 = datetime.datetime.now()

    yelp_reviews['Preprocessed_Text'] = [" ".join(review) for review in yelp_reviews['Preprocessed_Text'].values]
    #balanced_x, balanced_y = balance_classes(yelp_reviews['Preprocessed_Text'], yelp_reviews['stars'])
    #yelp_reviews['stars'] = [" ".join(review) for review in yelp_reviews['stars']]

    pipeline = Pipeline([('vct', TfidfVectorizer(min_df=.0025, max_df=.1, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')),
                         ('chi', SelectKBest(chi2, k=1000)),
                         ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])

    X_train, X_test, y_train, y_test = train_test_split(yelp_reviews['Preprocessed_Text'], yelp_reviews['stars'], test_size=0.2)

    model = pipeline.fit(X_train, y_train)

    vectorizer = model.named_steps['vct']
    chi = model.named_steps['chi']
    classifier = model.named_steps['clf']

    feature_names = vectorizer.get_feature_names()
    feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
    feature_names = np.asarray(feature_names)

    target_names = ['1', '2',  '4', '5']
    print("top 10 keywords per class: ")
    for i, label in enumerate(target_names):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (label, " ".join(feature_names[top10])))

    print("accuracy score: "+str(model.score(X_test,y_test)))

    print(model.predict(['that was an awesome place. Great food!!!']))

    #vectors = vectorizer.fit_transform(yelp_reviews['Preprocessed_Text'])
    #print(datetime.datetime.now() - t1)

    #print(vectors.toarray())
    #Split the dataset to train and test sub-datasets


    #print("LinearSVC:")
    #classifier_LinearSVC(X_train, X_test, y_train, y_test)

    #print("classifier_MultinomialNB:")
    #classifier_MultinomialNB(X_train, X_test, y_train, y_test)

main()