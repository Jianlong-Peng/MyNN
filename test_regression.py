import numpy as np
import matplotlib.pyplot as plt
from layer import *
from nn import NeuralNetwork
from validation import reg_error,reg_mae,reg_rmse,reg_r2
from functions import f_sigmoid
import sys


def func1(X,beta):
    y = np.zeros(X.shape[0])
    for i in xrange(X.shape[0]):
        for j in xrange(X.shape[1]):
            y[i] += pow(X[i][j],beta[j][0])
        y[i] += beta[-1][0]
    return y.reshape(X.shape[0],1)

def func2(X,beta):
    X2 = np.concatenate((np.power(X,2),np.ones(X.shape[0]).reshape(X.shape[0],1)),axis=1)
    y = X2.dot(beta)
    return y

def func3(X,beta):
    X2 = np.concatenate((X,np.ones(X.shape[0]).reshape(X.shape[0],1)),axis=1)
    y = X2.dot(beta)
    return y


def test_regression(trainX,trainY,beta,testX,testY,verbose,model):
    print "number of parameters:",model.num_parameters()
    print "number of samples:",trainX.shape[0]
    print "number of independent variables:",trainX.shape[1]
    print "to train the model..."
    model.train(trainX,trainY,verbose=verbose)

    print "\napply model to training set"
    train_pred = model.predict(trainX)
    mae = reg_mae(trainY,train_pred)
    rmse = reg_rmse(trainY,train_pred)
    r2 = reg_r2(trainY,train_pred)
    print "mae=%g, rmse=%g, r2=%g"%(mae,rmse,r2)
    plt.plot(trainY,train_pred,'o',color='r',label="training set")

    print "\napply model to test set"
    test_pred = model.predict(testX)
    mae = reg_mae(testY,test_pred)
    rmse = reg_rmse(testY,test_pred)
    r2 = reg_r2(testY,test_pred)
    print "mae=%g, rmse=%g, r2=%g"%(mae,rmse,r2)
    print "np.mean(np.abs(testY))=%g"%(np.mean(np.abs(testY)))
    
    plt.plot(testY,test_pred,'o',color='g',alpha=0.5,label="test set")
    plt.xlabel("actualY")
    plt.ylabel("predictY")
    plt.legend(loc="lower right")
    xmin,xmax = plt.xlim()
    ymin,ymax = plt.ylim()
    _min = min(xmin,ymin)
    _max = max(xmax,ymax)
    plt.xlim(_min,_max)
    plt.ylim(_min,_max)
    plt.show()

    temp = (testY-test_pred).flatten()
    plt.plot(temp,'o',color='r',alpha=0.75,label='residual')
    plt.show()


if __name__ == "__main__":
    argv = sys.argv
    verbose = False
    if len(argv)==2 and argv[1]=="--verbose":
        verbose = True

    p = 10
    n = 100

    """
    print "\n=============================="
    print "test multivariate regression"
    trainX = np.random.multivariate_normal(np.zeros(p),np.eye(p),(n,))
    beta = np.random.randn(p+1).reshape(p+1,1)
    trainY = func3(trainX, beta) + np.random.normal(size=n).reshape(n,1)
    testX = np.random.multivariate_normal(np.zeros(p),np.eye(p),(500,))
    testY = func3(testX, beta)
    _in = LinearLayer(p)
    hidden1 = SigmoidLayer(4)
    hidden2 = SigmoidLayer(4)
    output = LinearLayer(1)
    model = NeuralNetwork(_in, hidden1, hidden2, output, bias=True)
    test_regression(trainX,trainY,beta,testX,testY,verbose,model)
    """

    print "\n=============================="
    print "test logistic regression"
    trainX = np.random.multivariate_normal(np.zeros(p),np.eye(p),(n,))
    beta = np.random.randn(p+1).reshape(p+1,1)
    trainY = f_sigmoid(func3(trainX,beta)) + np.random.normal(0,0.1,n).reshape(n,1)
    testX = np.random.multivariate_normal(np.zeros(p),np.eye(p),(500,))
    testY = f_sigmoid(func3(testX,beta))
    _in = LinearLayer(p)
    hidden1 = SigmoidLayer(4)
    #output  = SigmoidLayer(1)
    output = LinearLayer(1)
    model = NeuralNetwork(_in, hidden1, output, bias=True)
    test_regression(trainX,trainY,beta,testX,testY,verbose,model)


