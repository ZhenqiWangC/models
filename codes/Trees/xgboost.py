from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
# set printing options
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from sklearn.feature_selection import SelectKBest,chi2
import re
from nltk.corpus import stopwords
import glob
from sklearn.feature_extraction.text import TfidfVectorizer



def rf(x,y,n_estimators,max_depth):
    clf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    clf.fit(x, y)
    return clf

def tune_rf(x,y,alpha_list,n,scoring):
    scores=[]
    for alpha in alpha_list:
        print alpha
        clf = RandomForestClassifier(n_estimators=n,max_depth=alpha)
        clf.fit(x, y)
        scores.extend([np.mean(cross_val_score(clf, x, y, cv=5, scoring=scoring))])
    max_index = scores.index(min(scores))
    print scores
    return alpha_list[max_index]



def bst(x,y,n,alpha,rate):
    clf = GradientBoostingClassifier(n_estimators=n,max_depth=alpha,learning_rate=rate)
    clf.fit(x, y)
    return clf

def xgbt(x,y,test,n,alpha,rate):
    clf = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=rate,
                 max_depth=alpha,
                 min_child_weight=1.5,
                 n_estimators=n,
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)
    clf.fit(x, y)
    return clf

def tune_xgbt(x,y,test,alpha_list,n,rate,scoring):
    scores=[]
    for alpha in alpha_list:
        clf = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=rate,
                 max_depth=alpha,
                 min_child_weight=1.5,
                 n_estimators=n,
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)
        clf.fit(x, y)
        scores.extend([np.mean(cross_val_score(clf, x[variables], y, cv=5, scoring=scoring))])
    max_index = scores.index(min(scores))
    print scores
    return alpha_list[max_index]


def tune_bst(x,y,test,alpha_list,n,rate,scoring):
    scores=[]
    for alpha in alpha_list:
        clf = GradientBoostingClassifier(n_estimators=n,max_depth=alpha,learning_rate=rate)
        variables = list(set(list(x)).intersection(list(test)))
        clf.fit(x[variables], y)
        scores.extend([np.mean(cross_val_score(clf, x[variables], y, cv=5, scoring=scoring))])
    max_index = scores.index(min(scores))
    print scores
    return alpha_list[max_index]

def tune_regtree(x,y,alpha_list,scoring):
    scores=[]
    for alpha in alpha_list:
        clf = DecisionTreeRegressor(max_depth=alpha)
        clf.fit(x, y)
        scores.extend([np.mean(cross_val_score(clf, x, y, cv=5, scoring=scoring))])
    max_index = scores.index(min(scores))
    print scores
    return alpha_list[max_index]



def get_coef(estimator):
    int = estimator.intercept_
    coef = estimator.coef_
    return coef

def predict(estimator,test):
    y_pred = estimator.predict(test)
    return y_pred


def load_data_census():
    train =pd.read_csv("./hw5_census_dist/train_data.csv",delimiter=",")
    test=pd.read_csv("./hw5_census_dist/test_data.csv",delimiter=",")
    for col in ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week','label']:
        train[col]=train[col].astype('int')
    for col in ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']:
        test[col]=test[col].astype('int')
    train = train.fillna(train.median())
    test = test.fillna(test.median())
    # turn nominal variables to dummies
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)
    common_var=list(set(list(train)).intersection(list(test)))
    return train[common_var],train['label'],test[common_var],common_var


def load_txt(file):
    file = open(file, 'r')
    text = file.read().lower()
    file.close()
    text = text.replace('\r\n', ' ')
    text= re.sub("[^a-z$!:?</@#%([\*]", " ", text)
    #text.translate(None, string.punctuation)
    words = text.split()
    stop = set(stopwords.words('english'))
    words = [i for i in words if i not in stop]
    return " ".join(words)

def txt_to_array(file_match):
    text_array = []
    onlyfiles =glob.glob(file_match)
    for i in range(len(onlyfiles)):
        text_array.append(load_txt(onlyfiles[i]))
    return text_array

def test_to_array():
    text_array = []
    for i in range(10000):
        text_array.append(load_txt("dist/test/"+str(i)+".txt"))
    return text_array

if __name__ == "__main__":
    # printing options

    x,y,test,vars=load_data_census()

    np.set_printoptions(precision=4,suppress=True)
    rate=0.01
    """
    for n in [100,150,200]:
        print n
        print (tune_rf(x,y,[5,10,15,20,30],n,scoring='accuracy'))
    alpha = tune_rf(x,y,[5,10,15,20,30],n=100,scoring='accuracy')
    print x.shape,test.shape
    estimator = rf(x,y,50,alpha)
    pred=predict(estimator,test)
    np.savetxt("census1.csv",pred,delimiter=",")

    alpha = tune_bst(x,y,test,[5,10,15,20,30],n=2000,rate=rate,scoring='accuracy')
    estimator = bst(x,y,n=2000,rate=rate,alpha=4)
    pred=predict(estimator,test)
    np.savetxt("census1.csv",pred,delimiter=",")


    alpha = tune_xgbt(x,y,test,[4],n=7000,rate=rate,scoring='accuracy')
    print alpha
    vars,estimator = xgbt(x,y,test,n=7000,rate=rate,alpha=alpha)
    y_pred=predict(estimator,x[vars],"log")
    plot_resy(y_pred,response)
    predict_test(estimator,test[vars],"log")
    """


    ham_x = txt_to_array("dist/ham/*.txt")
    spam_x = txt_to_array("dist/spam/*.txt")
    test_x = test_to_array()
    label = np.zeros(len(ham_x))
    label = np.concatenate((label,np.ones(len(spam_x))))
    x = np.concatenate((ham_x,spam_x))
    vectorizer = TfidfVectorizer(ngram_range = (1,2),analyzer="word", lowercase=False)
    vectorizer = vectorizer.fit(x.tolist())
    train_data_features = vectorizer.transform(x.tolist())
    fselect = SelectKBest(chi2 , k=500)
    train_data_features = fselect.fit_transform(train_data_features,label)
    train_data_features = train_data_features.toarray()
    test = vectorizer.transform(test_x)
    test = fselect.transform(test)
    test = test.toarray()


    #for n in [300,400,500,600]:
    print (tune_rf(train_data_features,label ,[15,25,30],200,scoring='accuracy'))
    alpha = tune_rf(train_data_features,label ,[15,25,30],n=200,scoring='accuracy')
    print x.shape,test.shape
    estimator = rf(train_data_features,label ,200,25)
    pred=predict(estimator,test)
    np.savetxt("spam1.csv",pred,delimiter=",")

