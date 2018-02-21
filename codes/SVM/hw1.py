from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.io
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest,chi2
import re
import string
from nltk.corpus import stopwords
import glob
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data_spam():
    x = scipy.io.loadmat("hw01_data/spam/spam_data.mat")
    trainx = x['training_data']
    trainy = x['training_labels']
    trainx = preprocessing.scale(trainx, axis=0, with_mean=False, with_std=True, copy=True)
    test = preprocessing.scale(x['test_data'], axis=0, with_mean=False, with_std=True, copy=True)
    trainy = trainy.reshape(trainy.shape[1],trainy.shape[0]).ravel()
    # shuffle training datasets
    np.random.seed(500)
    position = np.random.permutation(trainx.shape[0]).tolist()
    trainx = trainx[position,:]
    trainy = trainy[position]
    return trainx,trainy,test

def load_data_mnist():
    data = scipy.io.loadmat("hw01_data/mnist/train.mat")
    data = data['trainX']
    test = scipy.io.loadmat("hw01_data/mnist/test.mat")
    test = test['testX']
    trainx = data[:,:data.shape[1]-1]
    trainy = data[:,data.shape[1]-1]
    # shuffle training datasets
    np.random.seed(500)
    position = np.random.permutation(trainx.shape[0]).tolist()
    trainx = preprocessing.scale(trainx[position,:], axis=0, with_mean=True, with_std=True, copy=True)
    test = preprocessing.scale(test, axis=0, with_mean=True, with_std=True, copy=True)
    trainy = trainy[position]
    return trainx,trainy,test

def load_data_CIFAR():
    data = scipy.io.loadmat("hw01_data/cifar/train.mat")
    data = data['trainX']
    test = scipy.io.loadmat("hw01_data/cifar/test.mat")
    test = test['testX']
    trainx = data[:,:data.shape[1]-1]
    trainy = data[:,data.shape[1]-1]
    # shuffle training datasets
    np.random.seed(500)
    position = np.random.permutation(trainx.shape[0]).tolist()
    trainx = trainx[position,:]
    trainy = trainy[position]
    return trainx,trainy,test

def train_val_error(train_size,X_train,y_train,X_test,y_test):
    train_error = []
    val_error = []
    for i in train_size:
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train,y_train,train_size=i, random_state=20)
        model = SVC(kernel='linear')
        model.fit(X_train1,y_train1)
        train_error.append(1- accuracy_score(y_train1, model.predict(X_train1)))
        val_error.append(1- accuracy_score(y_test, model.predict(X_test)))
    return train_error,val_error

def plot_error(train_size,train_error,test_error,title):
    a= plt.scatter(train_size,train_error, color='b')
    b= plt.scatter(train_size,test_error, color='r')
    plt.title(title)
    plt.xlabel('Training Size')
    plt.ylabel('Error Rate')
    plt.legend((a,b),('Training Error', 'Validation Error'),
           scatterpoints=1,loc='upper right',ncol=3,fontsize=13)
    plt.show()

def tune_mnist(X_train,y_train,X_test,y_test,alpha_list):
    val_accuracy=[]
    for alpha in alpha_list:
        model = SVC(C=alpha)
        model.fit(X_train,y_train)
        val_accuracy.append(accuracy_score(y_test, model.predict(X_test)))
        print accuracy_score(y_test, model.predict(X_test))
    max_index =  val_accuracy.index(max(val_accuracy))
    print "val_error:", val_accuracy
    print "Best C:",alpha_list[max_index]
    return alpha_list[max_index]

def tune_CIFAR(X_train,y_train,X_test,y_test,alpha_list):
    val_accuracy=[]
    for alpha in alpha_list:
        model = SVC(C=alpha,kernel='linear')
        model.fit(X_train,y_train)
        val_accuracy.append(accuracy_score(y_test, model.predict(X_test)))
        print accuracy_score(y_test, model.predict(X_test))
    max_index =  val_accuracy.index(max(val_accuracy))
    print "val_error:", val_accuracy
    print "Best C:",alpha_list[max_index]
    return alpha_list[max_index]

def tune_spam(X_train,y_train,alpha_list):
    val_accuracy=[]
    for alpha in alpha_list:
        model = SVC(C=alpha)
        val_accuracy.extend([np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy'))])
        print [np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy'))]
    max_index =  val_accuracy.index(max( val_accuracy))
    print "CV_val_error:", val_accuracy
    print "Best C:",alpha_list[max_index]
    return alpha_list[max_index]

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
    for i in range(5857):
        text_array.append(load_txt("hw01_data/spam/test/"+str(i)+".txt"))
    return text_array


if __name__ == "__main__":

    """  Problem 2.1: MNIST """
    train,label,test = load_data_mnist()
    X_train, X_test, y_train, y_test = train_test_split(train,label,test_size=10000, random_state=10)
    size = [100, 200, 500, 1000, 2000, 5000,10000]
    train_error,val_error = train_val_error(size,X_train,y_train,X_test,y_test)
    plot_error(size,train_error,val_error,"MNIST SVM")

    """  Problem 3.1: MNIST """
    X_train1, X_test1, y_train1, y_test1 = train_test_split(train,label,train_size=15000, random_state=20)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_test1,y_test1,train_size=15000, random_state=10)
    X_train3, X_train4, y_train3, y_train4 = train_test_split(X_test2,y_test2,train_size=15000, random_state=30)
    #parameters = [0.00001,0.0001,0.001,0.01,0.1,10,1000,100000]
    #CV_val_error: 0.6822, 0.898, 0.9248,0.9255,0.9115,0.9068,0.9068,0.9068
    #parameters = [0.005,0.007,0.009,0.01,0.015,0.03]
    #CV_val_error: [0.927,0.9272,0.9266,0.9255,0.924,0.921]
    parameters = [0.005,0.007,0.009,0.01,0.015,0.03]
    best_c =  tune_mnist(X_train1,y_train1,X_test,y_test,parameters)
    model = SVC(C=best_c)
    model.fit(train,label)
    y_pred = model.predict(test)
    np.savetxt("mnist_predict.csv",y_pred,delimiter=",")

    """  Problem 2.2: Spam Emails """
    train,label,test = load_data_spam()
    X_train, X_test, y_train, y_test = train_test_split(train,label,train_size=0.8, random_state=10)
    """ set 80% as training set and 20% as validation set"""
    size = [100, 200, 500, 1000, 2000,X_train.shape[0]-1]
    train_error,val_error = train_val_error(size,X_train,y_train,X_test,y_test)
    plot_error(size,train_error,val_error,"Spam SVM")

    """ Problem 4: Spam Tuning C """
    #parameters = [0.00001,0.0001,0.001,0.01,0.1,10,100,1000]
    #CV_val_error: [0.7099, 0.7099, 0.7099, 0.7099, 0.7848, 0.8300, 0.8306, 0.8267, 0.8265]
    parameters= [50,80,100,150,200,250,300,500,800]
    #CV_val_error: [0.8236, 0.8246, 0.8257, 0.8242, 0.8250, 0.8250, 0.8238, 0.8223, 0.8213]
    best_c = tune_spam(train,label,parameters)
    model = SVC(C=best_c)
    model.fit(train,label)
    y_pred = model.predict(test)
    np.savetxt("spam_predict.csv",y_pred,delimiter=",")

    """ Problem 5: Spam kaggle """
    ham_x = txt_to_array("hw01_data/spam/ham/*.txt")
    spam_x = txt_to_array("hw01_data/spam/spam/*.txt")
    test_x = test_to_array()
    label = np.zeros(len(ham_x))
    label = np.concatenate((label,np.ones(len(spam_x))))
    x = np.concatenate((ham_x,spam_x))
    vectorizer = TfidfVectorizer(ngram_range = (1,2),analyzer="word", lowercase=False)
    vectorizer = vectorizer.fit(x.tolist())
    train_data_features = vectorizer.transform(x.tolist())
    fselect = SelectKBest(chi2 , k=2000)
    train_data_features = fselect.fit_transform(train_data_features,label)
    train_data_features = train_data_features.toarray()
    test = vectorizer.transform(test_x)
    test = fselect.transform(test)
    test = test.toarray()
    parameters= [7000,7200,7500,7700,8000]
    #CV_val_error: [0.9659, 0.9661, 0.9659, 0.9659, 0.9655]
    best_c = tune_spam(train_data_features,label,parameters)
    model = SVC(C=best_c)
    model.fit(train_data_features,label)
    y_pred = model.predict(test)
    np.savetxt("spam_predict.csv",y_pred,delimiter=",")


    """  Problem 2.3: CIFAR"""
    train,label,test = load_data_CIFAR()
    X_train, X_test, y_train, y_test = train_test_split(train,label,test_size=5000, random_state=10)
    size = [100, 200, 500, 1000, 2000,5000]
    train_error,val_error = train_val_error(size,X_train,y_train,X_test,y_test)
    plot_error(size,train_error,val_error,"CIFAR SVM")

    """  Problem 3.3: Tuning CIFAR"""
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train,y_train,train_size=5000, random_state=20)
    #parameters = [0.00001,0.0001,0.001,0.01,0.1,10,1000,100000]
    #val_error: [0.3018,0.292,0.292,0.292,0.292]
    #parameters = [0.000001,0.000005,0.00001,0.00001]
    #val_error:[0.3484,0.312,0.3018,0.3018]
    #parameters = [0.0000001,0.000001,0.000002]
    #val_error: [0.36699999999999999, 0.34839999999999999, 0.3306]
    #parameters = [0.000000001,0.0000001,0.0000005]
    #val_error: [0.22720000000000001, 0.36699999999999999, 0.36080000000000001]
    parameters = [0.000000005,0.000000008,0.0000001,0.0000002]
    #val_error: [0.32979999999999998, 0.3422, 0.36699999999999999, 0.36480000000000001]
    best_c = tune_mnist(X_train1,y_train1,X_test,y_test,parameters)
    model = SVC(C=best_c, kernel='linear')
    model.fit(train,label)
    y_pred = model.predict(test)
    np.savetxt("CIFAR_predict.csv",y_pred,delimiter=",")
