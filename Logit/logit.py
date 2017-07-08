from __future__ import division
import numpy as np
from numpy import random
import math
import matplotlib.pyplot as plt
import scipy.io
from numpy import genfromtxt

# 1. Gradient Descent of Logistic Regression
def GD_logit(X,y,reg=0.1,alpha=0.1,iteration=10):
    A=np.dot(X.transpose(),y)
    beta=np.zeros(X.shape[1],dtype="float")
    loss=np.zeros(iteration,dtype="float")
    for iter in range(iteration):
        u=np.zeros(X.shape[0],dtype="float")
        for s in range(len(X)):
                u[s]=1/(1+math.exp(-np.dot(beta,X[s])))
        B=np.dot(X.transpose(),u)
        gradient=2*reg*np.concatenate((beta[0:(len(beta)-1)],[0]))-A+B
        beta=beta-gradient*alpha/(len(X))
        u=np.zeros(X.shape[0],dtype="float")
        for s in range(len(X)):
                u[s]=1/(1+math.exp(-np.dot(beta,X[s])))
        loss[iter]=reg*np.dot(beta,beta)-np.dot(y,np.log(u))-np.dot(1-y,np.log(1-u))
    return loss,beta

# 2. Stochastic Gradient Descent:
def SGD_logit(X,y,reg=1,alpha=0.01,iteration=10):
    A=np.zeros((X.shape[0],X.shape[1]),dtype="float")
    for i in range(len(X)):
        A[i]=y[i]*X[i]
    beta=np.zeros(X.shape[1],dtype="float")
    loss=np.zeros(iteration,dtype="float")
    for iter in range(iteration):
        position=np.random.randint(0,len(y)-1)
        u=1/(1+math.exp(-np.dot(beta,X[position])))
        B=u*X[position]
        gradient=2*reg*np.concatenate((beta[0:(len(beta)-1)],[0]))-A[position]+B
        beta=beta-gradient*alpha/len(X)
        u=np.zeros(X.shape[0],dtype="float")
        for s in range(len(X)):
                u[s]=1/(1+math.exp(-np.dot(beta,X[s])))
        loss[iter]=reg*np.dot(beta,beta)-np.dot(y,np.log(u))-np.dot(1-y,np.log(1-u))
    return loss,beta

# 3. Stochastic Gradient Descent with non-constant learning rate
def SGD_logit_1(X,y,reg=1,alpha=0.01,iteration=10):
    A=np.zeros((X.shape[0],X.shape[1]),dtype="float")
    for i in range(len(X)):
        A[i]=y[i]*X[i]
    beta=np.zeros(X.shape[1],dtype="float")
    loss=np.zeros(iteration,dtype="float")
    for iter in range(iteration):
        position=np.random.randint(0,len(y)-1)
        u=1/(1+math.exp(-np.dot(beta,X[position])))
        B=u*X[position]
        gradient=2*reg*np.concatenate((beta[0:(len(beta)-1)],[0]))-A[position]+B
        beta=beta-gradient*alpha/float(iter+1)
        u=np.zeros(X.shape[0],dtype="float")
        for s in range(len(X)):
                u[s]=1/(1+math.exp(-np.dot(beta,X[s])))
        loss[iter]=reg*np.dot(beta,beta)-np.dot(y,np.log(u))-np.dot(1-y,np.log(1-u))
    return loss,beta

def test_accuracy(X,y,beta):
    u=np.zeros(X.shape[0], dtype="float")
    for s in range(len(X)):
            u[s]=1/(1+math.exp(-np.dot(beta,X[s])))
    pred=[round(t,0) for t in u ]
    accuracy=np.sum(1*np.array(pred==y,dtype="bool"),dtype='float')/float(len(u))
    return accuracy

def predict(X,beta):
    u=np.zeros(X.shape[0], dtype="float")
    for s in range(len(X)):
            u[s]=1/(1+math.exp(-np.dot(beta,X[s])))
    pred=[round(t,0) for t in u ]
    return pred

# 4. Use cross validation to tune regularization parameter
def tune_reg(X,y,reglist,cvfold=5,iteration=500,file="tune.csv"):
    max_accuracy=0
    position = np.random.permutation(len(y)).tolist()
    # start cross validation to assess Lambda/penalty performance
    size = int(len(y)/cvfold)
    accuracy=np.zeros(cvfold,dtype="float")
    max_accuracy=0
    for reg in reglist:
        print reg
        for i in range(cvfold):
            if i <> cvfold-1:
                x_cv = X[position[size*(i):size*(i+1)]]
                y_cv = y[position[size*(i):size*(i+1)]]
                t = position[0:size*(i)]+position[size*(i+1):]
                x_tr = X[t]
                y_tr = y[t]
            else:
                # cross-validation set
                x_cv = X[position[size*i:]]
                y_cv = y[position[size*i:]]
                # training set
                x_tr = X[position[0:size*i]]
                y_tr = y[position[0:size*i]]
            loss,beta=GD_logit(x_tr,y_tr,reg=reg,alpha=0.05,iteration=iteration)
            accuracy[i]=test_accuracy(x_cv,y_cv,beta)
        print np.mean(accuracy)
        if np.mean(accuracy)>=max_accuracy:
            max_accuracy=np.mean(accuracy)
            bestreg=reg
            np.savetxt(file,loss,delimiter=',')
            f_handle = open(file, 'a')
            f_handle.write("reg="+str(bestreg))
            f_handle.close()
    return bestreg

if __name__ == "__main__":
    # load data
    matfile = scipy.io.loadmat('data.mat', squeeze_me=True, struct_as_record=False)
    Xtrain=matfile["X"]
    ytrain=matfile["y"]
    Xtest=matfile["X_test"]
    # preprocess method 1: standardize the columns
    Xtrain1 = (Xtrain - np.mean(Xtrain, axis=0)) / np.std(Xtrain, axis=0)
    Xtrain1 = np.hstack((Xtrain1,np.ones((Xtrain1.shape[0],1),float)))
    Xtest = (Xtest - np.mean(Xtest, axis=0)) / np.std(Xtest, axis=0)
    Xtest = np.hstack((Xtest,np.ones((Xtest.shape[0],1),float)))
    print Xtrain1[0]
    
    # 1. GD with constant learning rate
    loss,beta=GD_logit(Xtrain,ytrain,reg=1,alpha=0.001,iteration=5000)
    np.savetxt("GD1.CSV",loss,delimiter=',')
    GD1=genfromtxt("GD1.CSV",delimiter=",")
    plt.plot(GD1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Logistic Regression: Gradient Descent')
    plt.legend()
    plt.axis([0, 5000, 0, 3000])
    plt.show()

    # 2. SGD with constant learning rate
    loss,beta=SGD_logit(Xtrain,ytrain,reg=0.1,alpha=0.1,iteration=5000)
    np.savetxt("SGD1.csv",loss,delimiter=",")
    SGD1=genfromtxt("SGD1.CSV",delimiter=",")
    plt.plot(SGD1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Logistic Regression: Stochastic Gradient Descent')
    plt.legend()
    plt.axis([0, 5000, 0, 4000])
    plt.show()


    # 3. Stochastic GD with varying alpha
    loss,beta=SGD_logit_1(Xtrain,ytrain,reg=0.1,alpha=0.01,iteration=5000)
    np.savetxt("SGD_11.csv",loss,delimiter=",")


    SGD11=genfromtxt("SGD_11.CSV",delimiter=",")
    GD1=genfromtxt("SGD1.CSV",delimiter=",")
    plt.plot(SGD11,label="Decreasing Learning Rate")
    plt.plot(GD1,label="Constant Learning Rate")
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Logistic Regression: Stochastic Gradient Descent')
    plt.legend()
    plt.axis([0, 5000, 0, 4000])
    plt.show()


    loss,beta=GD_logit(Xtrain1,ytrain,reg=1,alpha=0.1,iteration=10000)
    np.savetxt("GD01.CSV",loss,delimiter=',')
    loss,beta=GD_logit(Xtrain1,ytrain,reg=1,alpha=0.01,iteration=10000)
    np.savetxt("GD02.CSV",loss,delimiter=',')
    loss,beta=GD_logit(Xtrain1,ytrain,reg=1,alpha=0.05,iteration=10000)
    np.savetxt("GD03.CSV",loss,delimiter=',')

        # plot loss per interation
    GD1=genfromtxt("GD01.CSV",delimiter=",")
    GD2=genfromtxt("GD02.CSV",delimiter=",")
    GD3=genfromtxt("GD03.CSV",delimiter=",")
    plt.plot(GD1,label="alpha=0.1")
    plt.plot(GD2,label="alpha=0.01")
    plt.plot(GD3,label="alpha=0.05")
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.axis([0, 10000, 0, 3000])
    plt.show()




    print tune_reg(Xtrain1,ytrain,[0.1,1,10,100],cvfold=3,iteration=10000,file='tune2.csv')#

    beta=GD_logit(Xtrain1,ytrain,reg=1,alpha=0.01,iteration=10000)[1]
    pred=predict(Xtest,beta)
    np.savetxt("kaggle.csv",pred,delimiter=",")
    #kaggle  0.99597

