
from __future__ import division
import scipy.io
import numpy as np
import copy
from PIL import Image
import matplotlib.pyplot as plt
import random

def load_mnist():
    x = scipy.io.loadmat("./hw7_data/mnist_data/images.mat")
    train = normalize(x['images'])
    return train

def normalize(data):
    data = data.astype(float)
    for i in range(0,data.shape[2]):
        data[:,:,i] =data[:,:,i]/255.0
    return data

def kmeans(data,k,diff_initial):
    optimal_clusters = []
    for rand in range(diff_initial):
        dict_list=[]
        random.seed(rand)
        initial_index = np.random.choice(data.shape[2], k)
        for i in range(k):
            dict_list.append({'center':data[:,:,initial_index[i]],'member':[],'member_in':[],'within-dis':0})
        # initial assignment
        for i in range(data.shape[2]):
            distances = []
            for d in range(k):
                distances.append(np.linalg.norm(data[:,:,i]-dict_list[d]['center']))
            assign_cluster = distances.index(min(distances))
            #dict_list[assign_cluster]['member'].append({i:distances})
            dict_list[assign_cluster]['member_in'].append(i)
            dict_list[assign_cluster]['within-dis']+=distances[assign_cluster]
        # update center
        for i in range(k):
            new_center = np.mean(data[:,:,dict_list[i]['member_in']])
            dict_list[i]['center'] = new_center
        # repeated reassign until converge
        converged = False
        while not converged :
            for i in range(k):
                new_center = np.mean(data[:,:,dict_list[i]['member_in']],axis = 2)
                dict_list[i]['center'] = new_center
                dict_list[i]['old_index'] = copy.deepcopy(dict_list[i]['member_in'])
                dict_list[i]['member_in'] = []
                dict_list[i]['member']=[]
                dict_list[i]['within-dis']=0
            for i in range(data.shape[2]):
                distances = []
                for d in range(k):
                    distances.append(np.linalg.norm(data[:,:,i]-dict_list[d]['center']))
                    assign_cluster = distances.index(min(distances))
                    #dict_list[assign_cluster]['member'].append({i:distances})
                    dict_list[assign_cluster]['member_in'].append(i)
                    dict_list[assign_cluster]['within-dis']+=distances[assign_cluster]
            for i in range(k):
                if set(dict_list[i]['old_index']) == set(dict_list[i]['member_in']):
                    converged = True
                else:
                    converged = False
        if rand==0:
            total_dis = 0
            for i in range(k):
                    total_dis+=dict_list[i]['within-dis']
            optimal_clusters = dict_list
        else:
            sub_dis = 0
            for i in range(k):
                    sub_dis+=dict_list[i]['within-dis']
            if sub_dis < total_dis:
                total_dis = sub_dis
                optimal_clusters = dict_list
    return optimal_clusters

def plot(clusters):
    for i in range(len(clusters)):
        img = Image.fromarray(clusters[i]['center']*255)
        img.convert('RGB').save(str(len(clusters))+str(i)+"means.png")
        img.show()

def load_face():
    x = Image.open("./hw7_data/low-rank_data/face.jpg")
    col, row = x.size
    x = x.convert('LA')  # makes it greyscale
    y = np.asarray(x.getdata(), dtype=np.float64)
    y = y[:, 0].reshape((row, col))
    return y

def load_sky():
    x = Image.open("./hw7_data/low-rank_data/sky.jpg")
    col, row = x.size
    x = x.convert('LA')  # makes it greyscale
    y = np.asarray(x.getdata(), dtype=np.float64)
    y = y[:, 0].reshape((row, col))
    return y


def load_joke():
    x = scipy.io.loadmat("./hw7_data/joke_data/joke_train.mat")
    vali = np.genfromtxt("./hw7_data/joke_data/validation.txt", delimiter=",")
    test = np.genfromtxt("./hw7_data/joke_data/query.txt", delimiter=",").astype("int")
    return x['train'], vali, test


def low_rank(data, k, plot=False):
    U, s, V = np.linalg.svd(data, full_matrices=True)
    news = copy.deepcopy(s[:k])
    S = np.zeros((U.shape[1], V.shape[0]), dtype=float)
    S[:k, :k] = np.diag(news)
    approx = np.dot(U, np.dot(S, V))
    if plot == True:
        img = Image.fromarray(approx)
        img.convert('RGB').save(str(k) + "lowrank.png")
        img.show()
    return approx


def plot_mse(data, n):
    mse = np.zeros((n))
    for i in range(n):
        approx = low_rank(data, i + 1)
        mse[i] = np.linalg.norm(data - approx)
    plt.plot(mse, color='r', linewidth=2.0)
    plt.xlabel('ranks')
    plt.ylabel('mse')
    plt.title('MSE of different ranks')
    plt.grid(True)
    plt.show()
    return mse

def rating_recommend(new_data,epochs,reg):
    U, s, V = np.linalg.svd(new_data, full_matrices=False)
    mse = []
    accu = []
    train_mse=[]
    for iter in range(epochs):
        part2 = reg * np.diag(np.ones((new_data.shape[1])))
        for c in range(new_data.shape[1]):
            part2 += np.outer(V[c, :], V[c, :])
        for row in range(new_data.shape[0]):
            part1 = np.dot(V.T, new_data[row,:])
            u_update=np.linalg.solve(part2,part1)
            U[row, :] = u_update

        part2 = reg * np.diag(np.ones((new_data.shape[1])))
        for r in range(new_data.shape[0]):
            part2 += np.outer(U[r,], U[r,])

        for col in range(new_data.shape[1]):
            part1 = np.dot(U.T,new_data[:,col])
            v_update = np.linalg.solve(part2, part1)
            V[col,:] = v_update

        if (iter % 10 == 0):
            mod = np.dot(U, V.T)
            pred = predict_raw(mod, validation)
            pred0 = predict(mod, validation)
            accu = accuracy(pred0, validation)
            mse = np.sum([item ** 2 for item in pred - validation[:, 2]])
            pred = predict_raw(mod, train)
            pred0 = predict(mod, train)
            train_accu = accuracy(pred0, train)
            train_mse = np.sum([item ** 2 for item in pred - train[:, 2]])
            print "Loss:", mse, "Train Loss", train_mse
            print "Validation Accuracy:", accu,"Training Accuracy:",train_accu
    return np.dot(U, V.T),mse,accu,train_mse,train_accu


def predict(matrix, test):
    pred = np.array([matrix[test[:,0].astype("int") - 1, test[:,1].astype("int") - 1] > 0]).astype("int")
    return pred[0]


def predict_raw(matrix, test):
    pred = np.array([matrix[test[:,0].astype("int") - 1, test[:,1].astype("int") - 1]])
    return pred[0]

def predict_kaggle(matrix, test):
    pred = np.array([matrix[test[:,1].astype("int") - 1, test[:,2].astype("int") - 1] > 0]).astype("int")
    return pred[0]

def accuracy(pred,test0):
    test = copy.deepcopy(test0)
    test[test[:,2]>0,2] = 1
    test[test[:, 2] < 0, 2] =0
    return np.sum([1 for i,j in zip(pred.astype("int"), test[:,2].astype("int")) if i == j])/len(pred)

if __name__ == "__main__":

    ### problem 1
    data = load_mnist()
    clusters = kmeans(data,5,2)
    plot(clusters)
    clusters_10 = kmeans(data,10,2)
    plot(clusters_10)
    clusters_20 = kmeans(data, 20, 2)
    plot(clusters_20)


    ### problem 2
    data = load_face()
    low_rank(data,5,plot=True)
    low_rank(data,20,plot=True)
    low_rank(data,100,plot=True)
    face = plot_mse(data,100)

    data1 = load_sky()
    low_rank(data1,5,plot=True)
    low_rank(data1,20,plot=True)
    low_rank(data1,100,plot=True)
    sky = plot_mse(data1,100)


    plt.plot(sky,label="sky")
    plt.plot(face,label="face")
    plt.xlabel('ranks')
    plt.ylabel('mse')
    plt.title('MSE of sky and face')
    plt.legend()
    plt.grid(True)
    plt.show()


    U, s, V = np.linalg.svd(data, full_matrices=True)
    U, s1, V = np.linalg.svd(data1, full_matrices=True)
    plt.plot(s1[:10] / np.sum(s1), label="sky")
    plt.plot(s[:10] / np.sum(s), label="face")
    plt.xlabel('order of singular values')
    plt.ylabel('singular values / sum of singular values')
    plt.title('Counted Variation of sky and face')
    plt.grid(True)
    plt.legend()
    plt.show()

    ### problem 3
    data,validation,test = load_joke()
    new_data = copy.deepcopy(data)
    new_data[np.isnan(data)] = 0

    new = data.flatten('F')  ### column-wise
    joke = np.array([[i] * data.shape[0] for i in range(1, data.shape[1] + 1)]).astype("int").flatten()
    user = range(1, data.shape[0] + 1) * data.shape[1]
    train = np.vstack((user, joke, new)).T
    train = train[~np.isnan(train).any(axis=1)]

    mse = []
    train_mse = []
    accu = []
    train_accu= []
    U, s, V = np.linalg.svd(new_data,full_matrices=False)
    for rank in [2,5,10,20]:
        news = copy.deepcopy(s[:rank])
        S = np.zeros((V.shape[0], V.shape[0]), dtype=float)
        S[:rank, :rank] = np.diag(news)
        approx = np.dot(U, np.dot(S, V))
        pred = predict_raw(approx,validation)
        pred0 = predict(approx,validation)
        accu.append(accuracy(pred0,validation))
        mse.append(np.sum([item**2 for item in pred-validation[:,2]]))
        pred = predict_raw(approx, train)
        pred0 = predict(approx, train)
        train_accu.append(accuracy(pred0, train))
        train_mse.append(np.sum([item ** 2 for item in pred-train[:, 2]]))
    print mse,accu
    print train_mse,train_accu
    plt.plot([2,5,10,20],mse,color='r', linewidth=2.0)
    plt.xlabel('ranks')
    plt.ylabel('mse')
    plt.title('Validation MSE of different ranks')
    plt.grid(True)
    plt.show()
    plt.plot([2,5,10,20],accu, color='b', linewidth=2.0)
    plt.xlabel('ranks')
    plt.ylabel('accuracy')
    plt.title('Validation Accuracy of different ranks')
    plt.grid(True)
    plt.show()

    plt.plot([2, 5, 10, 20], train_mse, color='r', linewidth=2.0)
    plt.xlabel('ranks')
    plt.ylabel('mse')
    plt.title('Training MSE of different ranks')
    plt.grid(True)
    plt.show()
    plt.plot([2, 5, 10, 20], train_accu, color='b', linewidth=2.0)
    plt.xlabel('ranks')
    plt.ylabel('accuracy')
    plt.title('Training Accuracy of different ranks')
    plt.grid(True)
    plt.show()
    """
    [5677.1, 7388.0, 8520.3, 8911.9]
    [0.705, 0.715, 0.717, 0.686]
    [18441623.0, 16333384.4, 14165432.8, 11304007.4]
    [0.732, 0.764, 0.793, 0.828]
    """


    mod = rating_recommend(new_data,epochs=10,reg=35)[0]

    """
    lambda=0.05
    Loss: 2139.51357447
    Train
    Loss
    21185676.8452
    Validation
    Accuracy: 0.729268292683
    Training
    Accuracy: 0.998245160092
    lambda=10
    Loss: 2075.30933416 Train Loss 20343654.6946
    Validation Accuracy: 0.727913279133 Training Accuracy: 0.998171074223
    lambda=20
    Loss: 2015.69878194 Train Loss 19514171.2113
    Validation Accuracy: 0.727642276423 Training Accuracy: 0.99762482916
    lambda=30
    Loss: 1786.13646597 Train Loss 16946203.311
    Validation Accuracy: 0.730352303523 Training Accuracy: 0.992421126194
    lambda=40
    Loss: 1645.50975324 Train Loss 13273678.6722
    Validation Accuracy: 0.729268292683 Training Accuracy: 0.983304141289
    lambda = 35
    Loss: 1698.29819995 Train Loss 15163897.9013
    Validation Accuracy: 0.731436314363 Training Accuracy: 0.988385104981   
    """

    mod = np.dot(U, V.T)
    pred = predict_kaggle(mod,test)
    np.savetxt("kaggle.csv", pred, delimiter=",")




    # 26:-5, 56:5, 58:-10, 100:-10, 19:5
    lee_ratings = np.zeros((100))
    lee_ratings[18] = 5.0
    lee_ratings[25] = -5.0
    lee_ratings[55] = 5.0
    lee_ratings[57] = -10
    lee_ratings[99] = -10

    joke = np.array([[i] for i in range(1, data.shape[1] + 1)]).astype("int").flatten()
    user = [1] * data.shape[1]
    lee = np.vstack((user, joke, lee_ratings)).T
    lee = lee[~np.isnan(lee).any(axis=1)]

    print lee

    mod = np.dot(U, V.T)
    pred = predict_kaggle(mod, lee)
    # the highest 5
    print pred.argsort()[-10:][::-1]
    # the lowest 5
    print pred.argsort()[:10][::-1]
    """
    # 26:-5, 56:5, 58:-10, 100:-10, 19:5
    50,43,30,31,32     50:8 43:5 30:1 31:5 32:2
    64,34,78,36,37      64:-10 34:-1 78:1 36:2 37:1
    """

