from __future__ import division
import sklearn.metrics as metrics
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import scipy.io
from PIL import Image

NUM_CLASSES = 26


def load_data():
    x = scipy.io.loadmat("hw6_data_dist/letters_data.mat")
    # list(x.keys())
    trainx = x['train_x']
    trainy = x['train_y']
    test = x['test_x']
    return trainx, trainy, test

# convert from lables to one-zero vectors
def one_hot(labels_train):
    y_train=np.zeros((labels_train.shape[0], NUM_CLASSES))
    for i in range(labels_train.shape[0]):
        y_train[i][labels_train[i][0]-1]=1
    #row of Y_train: lable k converted to a vector with value 1 on index k position and 0 elsewhere'''
    return y_train

# normalize X to be in range[0,1]
# use min-max normalization
def mm_normalize(X,min=-1,max=1):
    newX=np.empty(X.shape)
    for i in range(X.shape[1]):
        if X[:,i].min()<>X[:,i].max():
            newX[:,i]=(max-min)*(X[:,i]-X[:,i].min())/(X[:,i].max()-X[:,i].min())+min
        else:
            newX[:,i]=0
    #newX=(max-min)*(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))+min
    return newX

def accuracy(predict,true):
    count=0
    for ii in range(len(predict)):
         if true[ii][predict[ii].tolist().index(max(predict[ii]))]==1:
            count = count+1
    return count/len(predict)

def plot(predict,true):
    correct = 0
    wrong =0
    for ii in range(len(predict)):
        if correct!=5 or wrong!=5:
            if true[ii][predict[ii].tolist().index(max(predict[ii]))] == 1 and correct!=5:
                print "predicted", predict[ii].tolist().index(max(predict[ii])), "should be", true[ii].tolist().index(max(true[ii]))
                image = X_train0[position[:size], :][ii].reshape((28, 28))
                img = Image.fromarray(image)
                img.save("predicted"+str(predict[ii].tolist().index(max(predict[ii])))+"should be"+str(true[ii].tolist().index(max(true[ii])))+'.png')
                img.show()
                correct+=1
            if true[ii][predict[ii].tolist().index(max(predict[ii]))] != 1 and wrong != 5:
                print "predicted", predict[ii].tolist().index(max(predict[ii])), "should be", true[ii].tolist().index(
                    max(true[ii]))

                image = X_train0[position[:size], :][ii].reshape((28, 28))
                img = Image.fromarray(image)
                img.save("predicted" + str(predict[ii].tolist().index(max(predict[ii]))) + "should be" + str(
                    true[ii].tolist().index(max(true[ii]))) + '.png')
                img.show()
                wrong+=1
        else:
            break



def predict_relu(X,W,V):
    #H: Hidden layer activated by ReLU
    H = (abs(np.dot(X,V.transpose()))+np.dot(X,V.transpose()))/2
    # H with bias term 1
    H = np.append(H,np.ones(H.shape[0])[...,None],axis=1)
    Z = np.exp(np.dot(H,W.transpose()))
    # Z: activated by softmax
    Z = Z / np.sum(Z)
    pred_y=[]
    for i in range(X.shape[0]):
        pred_y=pred_y+[Z[i].tolist().index(max(Z[i]))]
    return Z,pred_y

def predict_tanh(X,W,V):
    #H: Hidden layer activated by tanh
    H =  np.tanh(np.dot(X,V.transpose()))
    # H with bias term 1
    H = np.append(H,np.ones(H.shape[0])[...,None],axis=1)
    Z = np.exp(np.dot(H,W.transpose()))
    # Z: activated by softmax
    Z = Z / np.sum(Z)
    pred_y=[]
    for i in range(X.shape[0]):
        pred_y=pred_y+[Z[i].tolist().index(max(Z[i]))]
    return Z,pred_y


def train_NN_SGD_tanh(X,Y,d_hid,alpha1=0.001,alpha2=0.001,reg1=0.1,reg2=0.1,decay=0.8,epochs=10,plot=0):
    # shuffle data
    position = np.random.permutation(len(Y)).tolist()
    X=X[position,:]
    Y=Y[position,:]
    # to keep loss and accuracy while iterates
    loss=[]
    accurate=[]
    # generate random weight matrix by Gaussian(0,1)
    # V: input to hidden weight matrix
    V = 0.1*np.random.randn(d_hid,X.shape[1])
    # W: hidden to output weight matrix
    W = 0.1*np.random.randn(Y.shape[1],d_hid+1)
    alpha1_0=alpha1
    alpha2_0=alpha2
    # run through shuffled training dataset for SGD
    for t in range(epochs):
        print t
        alpha1_0 = alpha1_0 * decay
        alpha2_0 = alpha2_0 * decay
        for s in range(X.shape[0]):
            alpha1 = alpha1_0 * ((epochs * X.shape[0] - t * X.shape[0] - s) / (1.3*epochs * X.shape[0]))
            alpha2 = alpha2_0 * ((epochs * X.shape[0] - t * X.shape[0] - s) / (1.3*epochs * X.shape[0]))
            # compute all nodes
            # H_temp: X dot V.T
            H_temp = np.dot(X[s],V.transpose())
            #H: H_temp activated by tanh
            H0 = np.tanh(H_temp)
            # H with bias term 1
            H = np.append(H0,1)[...,None]
            temp = np.dot(H.transpose(),W.transpose())
            #print np.amin(temp),np.amax(temp)
            Z = np.exp(temp)
            # Z: activated by softmax
            Z = Z / np.sum(Z)
            # don't penalize intercept term: the last column of V
            V_pen = V[:,:(V.shape[1]-1)]
            V_pen = np.hstack((V_pen, np.reshape(np.zeros(V_pen.shape[0]), (V_pen.shape[0], 1))))
            part1 = list(np.power(H0,2))
            part1 = [1.0-ss for ss in part1]
            part2 = np.multiply(part1,np.dot(Z-Y[s],W[:,:d_hid]))
            part3 = np.outer(part2,X[s])
            V_gradient=part3+reg1*V_pen
            # don't penalize intercept term: the last column of W
            W_pen = W[:,:(W.shape[1]-1)]
            W_pen = np.hstack((W_pen, np.reshape(np.zeros(W_pen.shape[0]), (W_pen.shape[0], 1))))
            W_gradient=np.outer(Z-Y[s],H)+reg2*W_pen
            V=V-V_gradient*alpha1
            W=W-W_gradient*alpha2
            if (plot==1 and s%5000==0):
                results=predict_tanh(X,W,V)[0]
                loss.append(-np.sum(np.multiply(np.log(results),Y)))
                accurate.append(accuracy(results,Y))
                print "Loss:",loss[-1]
                print "Training Accuracy:",accurate[-1]
    return W,V,loss,accurate


def train_NN_SGD_relu(X,Y,d_hid,alpha1=0.001,alpha2=0.001,reg1=0.1,reg2=0.1,decay=1,epochs=10,plot=0):
    # shuffle data
    position = np.random.permutation(len(Y)).tolist()
    X=X[position,:]
    Y=Y[position,:]
    # to keep loss and accuracy while iterates
    loss=[]
    accurate=[]
    # generate random weight matrix by Gaussian(0,1)
    # V: input to hidden weight matrix
    V = 0.1*np.random.randn(d_hid,X.shape[1])
    # W: hidden to output weight matrix
    W = 0.1*np.random.randn(Y.shape[1],d_hid+1)
    alpha1_0=alpha1
    alpha2_0=alpha2
    # run through shuffled training dataset for SGD
    for t in range(epochs):
        print t
        alpha1_0=alpha1_0*decay
        alpha2_0=alpha2_0*decay
        for s in range(X.shape[0]):
            alpha1 = alpha1_0 * ((epochs*X.shape[0]-t*X.shape[0]-s)/(1.1*epochs*X.shape[0]))
            alpha2 = alpha2_0 * ((epochs*X.shape[0]-t*X.shape[0]-s)/(1.1*epochs*X.shape[0]))
            # compute all nodes
            # H_temp: X dot V.T
            H_temp = np.dot(X[s],V.transpose())
            #H: H_temp activated by ReLU
            H = (abs(H_temp)+H_temp)/2
            # H with bias term 1
            H = np.append(H,1)[...,None]
            temp = np.dot(H.transpose(),W.transpose())
            #print np.amin(temp),np.amax(temp)
            Z = np.exp(temp)
            # Z: activated by softmax
            Z = Z / np.sum(Z)
            # don't penalize intercept term: the last column of V
            V_pen = V[:,:(V.shape[1]-1)]
            V_pen = np.hstack((V_pen, np.reshape(np.zeros(V_pen.shape[0]), (V_pen.shape[0], 1))))
            V_gradient=np.outer(np.dot(Z-Y[s],W[:,0:d_hid]),X[s])+reg1*V_pen
            neg=np.where(H_temp<0)
            V_gradient[neg,:]=0
            # don't penalize intercept term: the last column of W
            W_pen = W[:,:(W.shape[1]-1)]
            W_pen = np.hstack((W_pen, np.reshape(np.zeros(W_pen.shape[0]), (W_pen.shape[0], 1))))
            W_gradient=np.outer(Z-Y[s],H)+reg2*W_pen
            V=V-V_gradient*alpha1
            W=W-W_gradient*alpha2
            if (plot==1 and s%5000==0):
                results=predict(X,W,V)[0]
                loss.append(-np.sum(np.multiply(np.log(results),Y)))
                accurate.append(accuracy(results,Y))
                print "Loss:",loss[-1]
                print "Training Accuracy:",accurate[-1]
    return W,V,loss,accurate


if __name__ == "__main__":
    X_train0, labels_train, X_test = load_data()
    # normalize and add bias term
    X_train = mm_normalize(X_train0,0,1)
    X = np.append(X_train,np.ones(X_train.shape[0])[...,None],axis=1)
    X_test = mm_normalize(X_test,0,1)
    X_test = np.append(X_test,np.ones(X_test.shape[0])[...,None],axis=1)
    Y = one_hot(labels_train)
    # shuffle the data and split training and validation set
    position = np.random.permutation(len(Y)).tolist()
    size = int(len(Y) /5)
    valx = X[position[:size],:]
    valy = Y[position[:size]]
    trainx = X[position[size:],:]
    trainy = Y[position[size:],:]

    image = X_train0[0].reshape((28,28))
    img = Image.fromarray(image)
    img.save('my.png')
    img.show()


    (W, V, loss, rates) = train_NN_SGD_tanh(trainx, trainy, 200, alpha1=0.02, alpha2=0.01, reg1=0.001, reg2=0.001, decay=0.7,
                                       epochs=2, plot=0)
    results = predict_tanh(trainx, W, V)[0]
    print "training accuracy:", accuracy(results,trainy)
    results = predict_tanh(valx, W, V)[0]
    print "validation accuracy:",accuracy(results,valy)

    # training accuracy: 0.875691105769
    # validation accuracy: 0.863100961538

    np.savetxt("loss.csv", loss, delimiter=",")
    np.savetxt("accuracy.csv", rates, delimiter=",")
    rates = np.loadtxt("accuracy.csv", delimiter=",")
    loss = np.loadtxt("loss.csv", delimiter=",")
    iteration = np.array(range(len(loss))) * 5000
    plt.plot(iteration[:100], loss[:100], color='r', linewidth=2.0)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss of Neural Network SGD')
    plt.grid(True)
    plt.show()
    plt.plot(iteration[:100], rates[:100], color='b', linewidth=2.0)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Learning accuracy of Neural Network SGD')
    plt.grid(True)
    plt.show()
    #training accuracy: 0.930158253205
    # val accuracy: 0.904326923077
    (W, V, loss, rates) = train_NN_SGD_relu(trainx, trainy, 1200, alpha1=0.05, alpha2=0.06, reg1=0.001 , reg2=0.0014, decay=0.8,
                                       epochs=1, plot=0)

    results = predict_relu(trainx, W, V)[0]
    print "training accuracy:", accuracy(results, trainy)
    results = predict_relu(valx, W, V)[0]
    print "validation accuracy:", accuracy(results, valy)
    plot(results,valy)


    np.savetxt("loss.csv", loss, delimiter=",")
    np.savetxt("accuracy.csv", rates, delimiter=",")
    rates = np.loadtxt("accuracy.csv", delimiter=",")
    loss = np.loadtxt("loss.csv", delimiter=",")
    iteration = np.array(range(len(loss))) * 5000
    plt.plot(iteration[:100], loss[:100], color='r', linewidth=2.0)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss of Neural Network SGD')
    plt.grid(True)
    plt.show()
    plt.plot(iteration[:100], rates[:100], color='b', linewidth=2.0)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Learning accuracy of Neural Network SGD')
    plt.grid(True)
    plt.show()
    (W, V, loss, rates) = train_NN_SGD_relu(X, Y, 1200, alpha1=0.05, alpha2=0.06, reg1=0.001, reg2=0.0014,
                                       decay=0.8,
                                       epochs=7, plot=0)
    np.savetxt("kaggle.csv", predict_relu(X_test, W, V)[1], delimiter=",")






