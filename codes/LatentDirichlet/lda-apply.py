from __future__ import division
import pandas as pd
import re
from nltk.corpus import stopwords
from gensim import corpora
import numpy as np
from sklearn.externals import joblib
from nltk.tag import pos_tag
from sklearn.linear_model import LinearRegression


def findTopic(lda,review, dictionary):
    raw = review.text.tolist()
    stop = set(stopwords.words('english'))
    text_array = []
    for i in range(len(raw)):
        text = raw[i].lower()
        text = text.replace('\r\n', ' ')
        text = re.sub("[^a-z0-9]", " ", text)
        # Tokenization segments a document into its atomic elements.
        words = text.split()
        # Stop words
        # Certain parts of English speech, like (for, or) or the word the are meaningless to a topic model.
        # These terms are called stop words and need to be removed from our token list.
        words = [j for j in words if j not in stop]
        tagged_sent = pos_tag(words)
        words = [word for word,pos in tagged_sent if pos == 'NN']
        text_array.append(words)
    all_topics=[]
    for text in text_array:
        doc_bow = dictionary.doc2bow(text)
        topics = [sorted(lda[doc_bow], key=lambda x: x[1], reverse=True)]
        all_topics.extend(topics)
    return all_topics

if __name__ == '__main__':

    # select target data
    data = pd.read_csv("./yelp_dataset_challenge_round9/yelp_academic_dataset_review.csv")
    state = "NC"
    bus = pd.read_csv("./data/" + str(state) + "business.csv")
    unique_bus_id = bus.business_id.unique()
    review = data[data['business_id'].isin(unique_bus_id)]
    filtered_id = review['business_id'].value_counts()
    selected_bus = bus[bus["business_id"] == "RAh9WCQAuocM7hYM5_6tnw"]
    review.to_csv("./data/" + "selected" + "bus.csv")
    review = data[data['business_id'] == "RAh9WCQAuocM7hYM5_6tnw"]
    review.to_csv("./data/" + "selected" + "reviews.csv")

    # load model and training dictionary
    filename = 'finalized_model_15_lda.sav'
    loaded_model = joblib.load(filename)
    dict = corpora.Dictionary()
    dict = dict.load('dictionary.dic')
    data = pd.read_csv("./data/" + "selected" + "reviews.csv")
    all_topics = findTopic(loaded_model,data, dict)


    topic_dict = {0:"Mexican",1:"Family",2:"Night/Bar",11:"Night/Bar",3:"Southeast_Asian",
                 4:"Dessert",5:"Japanese",6:"Breakfast",7:"Meat",8:"Dinner",9:"Service",
                  13:"Service",10:"American",12:"Ambiguous", 14:"Italian"}
    topic_count={"Mexican":0,"Family":0,"Night/Bar":0,"Southeast_Asian":0,
                 "Dessert":0,"Japanese":0,"Breakfast":0,"Meat":0,"Dinner":0,"Service":0,
                 "American":0,"Ambiguous":0, "Italian":0}

    # get training matrix for linear regression
    train = np.zeros((data.shape[0],15))
    for i in range(data.shape[0]):
        items = all_topics[i]
        for s in items:
            if s[1]>0.05:
                topic = topic_dict[s[0]]
                topic_count[topic]+=1
                train[i,s[0]] = 1/len(items)

    # select only high frequency subtopics
    # Family, Night/Bar,Dessert,Japanese,Meat,Dinner,Service,American
    train_new = train[:, [1, 2, 4, 5, 7, 8, 9, 10]]
    # merge similar topics togeterh
    train_new[:, 1] += train[:, 3]
    train_new[:, 6] += train[:, 13]
    y = data.stars
    # Create linear regression object
    regr = LinearRegression()
    # Train the model using the training sets
    regr.fit(train_new,y)
    # The coefficients
    print('Coefficients:', regr.coef_)
    print('Intercept:',regr.intercept_)
    # The mean squared error
    print("Mean squared error: %.2f" % np.mean((regr.predict(train_new) - y) ** 2))
    # Explained variance score: 1 is perfect prediction
    print ("ratings", regr.coef_ + regr.intercept_)
    """
    [3.3108761   4.23414281  4.82628184  4.46602252  4.63023886  4.30278387
     4.13174175  4.25038669]
    # ('Coefficients:', array([-0.79840283,  0.12486389,  0.71700291,  0.35674359,  0.52095993,
    0.19350494, 0.02246282, 0.14110777]))
    ('Intercept:', 4.1092789244842454)
    Mean
    squared
    error: 0.88
    Variance
    score: 0.01
    """

    # validatiaon
    x1 = train_new[:900]
    y1 = y[:900]
    xv = train_new[900:]
    yv = y[900:]
    regr = LinearRegression()
    # Train the model using the training sets
    regr.fit(x1,y1)
    # The coefficients
    print('Coefficients:', regr.coef_)
    print('Intercept:',regr.intercept_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % np.mean((regr.predict(xv) - yv) ** 2))