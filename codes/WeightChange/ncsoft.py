from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest,chi2
import re
from nltk.corpus import stopwords
import glob
from sklearn.feature_extraction.text import TfidfVectorizer

import itertools



def load_txt(file):
    file = open(file, 'r')
    text = file.read().lower()
    file.close()
    text = text.replace('\r\n', ' ')
    text= re.sub("[^a-z$!:?</@#%([\*]", " ", text)
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
def load_census():
    training = pd.read_csv('./hw5_census_dist/train_data.csv',na_values=["?"])
    test = pd.read_csv('./hw5_census_dist/test_data.csv',na_values=["?"])
    var_names = list(training)[:-1]
    y = training['label']
    x = training.ix[:,:-1]
    for col in var_names:
        if np.issubdtype(test[col].dtype, np.number):
            test[col].fillna(test[col].mean())
            x[col].fillna(x[col].mean())
        else:
            test[col].fillna(test[col].mode()[0])
            x[col].fillna(x[col].mode()[0])
    x = pd.get_dummies(x)
    test = pd.get_dummies(test)
    common_var = list(set(x).intersection(test))
    x = x[common_var]
    test = test[common_var]
    return np.array(x), np.array(y),test,common_var

def load_spam(k=2000):
    ham_x = txt_to_array("dist/ham/*.txt")
    spam_x = txt_to_array("dist/spam/*.txt")
    test_x = test_to_array()
    label = np.zeros(len(ham_x))
    label = np.concatenate((label, np.ones(len(spam_x))))
    x = np.concatenate((ham_x, spam_x))
    vectorizer = TfidfVectorizer(ngram_range=(1,1), analyzer="word", lowercase=False)
    vectorizer = vectorizer.fit(x.tolist())
    train_data_features = vectorizer.transform(x.tolist())
    fselect = SelectKBest(chi2, k=k)
    train_data_features = fselect.fit_transform(train_data_features, label)
    var_names = np.asarray(vectorizer.get_feature_names())[fselect.get_support()]
    train_data_features = train_data_features.toarray()
    test = vectorizer.transform(test_x)
    test = fselect.transform(test)
    test = test.toarray()
    return train_data_features,label,test,var_names

# 1. Gradient Descent of Logistic Regression
def GD_logit(X,y,reg=0.1,alpha=0.1,iteration=10):
    A=np.dot(X.transpose(),y)
    beta_matrix=np.zeros((X.shape[1],iteration),dtype="float")
    loss=np.zeros(iteration,dtype="float")
    beta = np.zeros(X.shape[1],dtype="float")
    for iter in range(iteration):
        beta_matrix[:,iter] = beta
        u=np.zeros(X.shape[0],dtype="float")
        for s in range(len(X)):
                u[s]=1/(1+np.exp(-np.dot(beta,X[s])))
        B=np.dot(X.transpose(),u)
        gradient=2*reg*np.concatenate((beta[0:(len(beta)-1)],[0]))-A+B
        beta=beta-gradient*alpha/(len(X))
        u=np.zeros(X.shape[0],dtype="float")
        for s in range(len(X)):
                u[s]=1/(1+np.exp(-np.dot(beta,X[s])))
        loss[iter]=reg*np.dot(beta,beta)-np.dot(y,np.log(u))-np.dot(1-y,np.log(1-u))
    return loss,beta_matrix

def test_accuracy(X,y,beta):
    u=np.zeros(X.shape[0], dtype="float")
    for s in range(len(X)):
            u[s]=1/(1+np.exp(-np.dot(beta,X[s])))
    pred=[round(t,0) for t in u ]
    accuracy=np.sum(1*np.array(pred==y,dtype="bool"),dtype='float')/float(len(u))
    return accuracy

def predict(X,beta):
    u=np.zeros(X.shape[0], dtype="float")
    for s in range(len(X)):
            u[s]=1/(1+np.exp(-np.dot(beta,X[s])))
    pred=[round(t,0) for t in u ]
    return pred

if __name__ == "__main__":
    ###### census #####
    x, y, test, vars = load_spam(k=50)
    position = np.random.permutation(x.shape[0]).tolist()
    x, y = x[position, :], y[position]
    data = np.concatenate((x, y.reshape((len(y), 1))), axis=1).tolist()
    training = data[:int(0.8 * len(data))]
    trainx = [row[:-1] for row in training]
    trainy = [row[-1] for row in training]
    valx = [row[:-1] for row in data[int(0.8 * len(data)):]]
    valy = [row[-1] for row in data[int(0.8 * len(data)):]]
    print vars
    loss, beta =  GD_logit(np.array(trainx),np.array(trainy), reg=1, alpha=10, iteration=1000)
    print test_accuracy(np.array(trainx),np.array(trainy),beta[:,-1])
    colormap = plt.cm.gist_ncar
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, x.shape[1])])
    marker = itertools.cycle(('o', 'v', '^', '<', '>', 's', '8', 'p'))
    s = np.arange(0,1000,100)
    fig = plt.figure(1)
    for i in np.arange(x.shape[1]):
        plt.plot(s,beta[i][s],linestyle='-', marker=marker.next() )
    lgd = plt.legend(vars,bbox_to_anchor=(1.05, 1), loc=2,
          ncol=2, fancybox=True, shadow=True,fontsize = 'x-small')
    plt.title("Weights for word frequencies in detecting Spam")
    plt.xlabel("iterations")
    plt.ylabel("weights")
    plt.show()
    fig.savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')
