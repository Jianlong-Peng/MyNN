'''
#=============================================================================
#     FileName: nn.py
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2015-02-05 10:25:09
#   LastChange: 2015-02-06 16:18:57
#      History:
#=============================================================================
'''
from random import random
import numpy as np
from layer import *

"""
3-layer feed-forward neural network

input layer :
  linear neuro
  in : X, i
hidden layer: Y, j
  activation: net_j = \sum_i{w_ij*x_i} + w_j
  out       : y_j = f(net_j)
output layer: Z, k
  activation: net_k = \sum_j{w_jk*y_j} + w_k
  out       : z_k = f(net_k)

cost function:
  J = 0.5*||T-Z||^2 + 0.5*lambda*||W||^2
  where, `T` is target, `Z` is predict

(1) for w_jk
  \partial{J}{w_jk} = \partial{J}{net_k} * \partial{net_k}{w_jk} + lambda*w_jk
  where, \partial{j}{net_k}
            = \partial{J}{z_k} * \partial{z_k}{net_k}
            = -1 * (t_k - z_k) * f'(net_k)
  let D_k = \partial{j}{net_k} ............................ (1)
  So, \partial{J}{w_jk} = D_k*y_j + lambda*w_jk ........... (2)
  therefore, the update rule of w_jk could be:
  w^{t+1}_jk = w^{t}_jk - rate * (D_k*y_j + lambda*w_jk)... (3)

(2) for w_ij
  \partial{J}{w_ij} = \partial{J}{y_j} * \partial{y_j}{net_j} * \partial{net_j}{w_ij} + lambda*w_ij
  where, \partial{J}{y_j}
            = -1 * \sum_k{(t_k-z_k) * \partial{z_k}{y_j}}
            = -1 * \sum_k{(t_k-z_k) * \partial{z_k}{net_k} * \partial{net_k}{y_j}}
            = -1 * \sum_k{(t_k-z_k) * f'(net_k) * w_jk}
  So, \partial{J}{w_ij}
         = -1 * \sum_k{(t_k-z_k)*f'(net_k)*w_jk} * f'(net_j) * x_i + lambda*w_ij
         = \sum_k{(t_k-z_k)*D_k} * f'(net_j) * x_i + lambda*w_ij
  let D_j = \sum_k{(t_k-z_k)*D_k}*f'(net_j) ............... (4)
  then, \partial{J}{w_ij} = D_j*x_i + lambda*w_ij.......... (5)
  therefore, the update rule of w_ij could be:
  w^{t+1}_ij = w^{t}_ij - rate * (D_j*x_i + lambda*w_ij)... (6)



More general cases (n-layer NN)
1. feed-forward part
   in layer `l` (l = 1 to L), 
   let input be `in_l` with shape (m, ni), weight be `w_l` with shape (ni, nj), and `in_0 = X`
   `net_l = in_l.dot(w_l)`     (m, nj)
   `out_l = f_l(net_l)`        (m, nj)
   `in_{l+1} = out_l`

2. back-propgation
   1) for the output layer
      D_L = (t-out_L)*f_L'(net_L)
      delta_L = in_L.T.dot(D_L)
   2) for hidden layers (l = 1 to L-1)
      D_l = D_{l+1}.dot(w_{l+1}.T) * f_l'(net_l)
      delta_l = in_l.T.dot(D_l)
      w_{l+1} = w_{l+1} + rate * (delta_l + lambda * w_{l+1})
"""

#Attention:
# 1. how to deal with NN with more than one output neuro???
# 2. pay attention to bias !!!!!! (BUG)
class NeuralNetWork:
    def __init__(self, *layers, **kargs):
        '''
        Parameters
        ==========
        layers: all layers
                the first layer is useless but to determine number of
                independent variables
        kargs :
                bias - 
        '''
        if len(layers) < 3:
            raise ValueError("there must be at least 3 layers, but only %d given!"%len(layers))
        self.layers = layers
        for l in self.layers:
            assert isinstance(l,Layer)
        self.bias = kargs.get("bias",False)
        n = 1 if self.bias else 0
        self.weight = [None for i in xrange(len(self.layers))]
        for i in xrange(len(self.layers)-1):
            self.weight[i+1] = \
                    np.asarray([[random() for j in xrange(self.layers[i].n+n)] for k in xrange(self.layers[i+1].n)]).T
            print self.weight[i+1].shape
    """
    def __init__(self, inputLayer, hiddenLayer, outputLayer, bias=True):
        self.inputLayer = inputLayer
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer
        self.bias = bias
        n = 1 if self.bias else 0
        self.wij = np.asarray([[random() for i in xrange(self.inputLayer.n+n)] for j in xrange(self.hiddenLayer.n)]).T
        self.wjk = np.asarray([[random() for j in xrange(self.hiddenLayer.n+n)] for k in xrange(self.outputLayer.n)]).T
    """

    def _copy_input(self, X):
        if self.bias:
            if X.shape[1] == self.layers[0].n:
                col = np.ones(X.shape[0]).reshape(X.shape[0],1)
                X = np.concatenate((X,col),axis=1)
            else:
                raise ValueError("invalid input dimension",X.shape)
        else:
            if X.shape[1] == self.layers[0].n:
                X = np.copy(X)
            else:
                raise ValueError("invalid input dimension",X.shape)
        return X

    def predict(self, X):
        '''
        Parameter
        =========
        X: np.ndarray with shape (m, ni)
           where, m is number of samples, ni is number of predictors

        Return
        ======
        np.ndarray with shape (m, nk)
        '''
        _in = self._copy_input(X)
        for i in xrange(1,len(self.layers)):
            _in = self.layers[i].activate(_in, self.weight[i])
            if self.bias:
                _in = np.concatenate((_in,np.ones(_in.shape[0]).reshape(_in.shape[0],1)),axis=1)
        if self.bias:
            return _in[:,:-1]
        else:
            return _in
    """
    def predict(self, X):
        X = self._copy_input(X)
        y = self.hiddenLayer.activate(X, self.wij)
        z = self.outputLayer.activate(y, self.wjk)
        return z
    """

    def train(self, X, y, _epoch=1E5, _epsilon=1E-3, _rate=1E-3, _lambda=0):
        '''
        Parameter
        =========
        X: np.ndarray with shape (m, ni)
           where, m is number of samples, ni is number of predictors
        y: np.ndarray with shape (m,)

        '''
        X = self._copy_input(X)
        if y.ndim == 1:
            t = y.reshape(t.shape[0],1)
        else:
            t = np.copy(y)

        n = 0
        err = 1E6
        while n < _epoch:
            n += 1
            #1. forward
            net = [None for _ in xrange(len(self.layers))]
            net[0] = X
            z = [None for _ in xrange(len(self.layers))]
            z[0] = X
            for i in xrange(1,len(self.layers)):
                net[i] = z[i-1].dot(self.weight[i])
                z[i] = self.layers[i].f(net[i])
                if self.bias:
                    z[i] = np.concatenate((z[i],np.ones(z[i].shape[0]).reshape(z[i].shape[0],1)),axis=1)
            for i in xrange(len(z)):
                print z[i].shape
            #2. estimate
            if self.bias:
                err = np.sum(np.abs(t-z[-1][:,:-1]))
            else:
                err = np.sum(np.abs(t-z[-1]))
            if err < _epsilon:
                break
            #3. backpropgation
            #error in the output layer
            D = (t-z[-1])*self.layers[-1].fprime(net[-1])
            print D.shape
            delta2 = z[-2].T.dot(D[-1])
            for i in xrange(len(net)-2, 0, -1):
                #error backpropgated from the last layer
                D = D.dot(self.weight[i+1].T) * self.layers[i].fprime(net[i])
                delta1 = z[i-1].T.dot(D)
                #update weight between the current and last layer
                self.weight[i+1] = self.weight[i+1] + _rate * (delta2 + _lambda * self.weight[i+1])
                delta2 = delta1
            self.weight[1] = self.weight[1] + _rate * (delta2 + _lambda * self.weight[1])

            """
            netj = X.dot(self.wij)                #m x nj
            yj   = self.hiddenLayer.f(netj)       #m x nj
            netk = yj.dot(self.wjk)               #m x nk
            z = self.outputLayer.f(netk)          #m x nk
            #2. estimate
            err = np.sum(np.abs(t-z))
            if err < _epsilon:
                break
            #3. backpropgation
            Dk = (t-z)*self.outputLayer.fprime(netk)                 #m x nk
            delta_jk = yj.T.dot(Dk)                                  #nj x nk
            Dj = Dk.dot(self.wjk.T) * self.hiddenLayer.fprime(netj)  #m x nj
            delta_ij = X.T.dot(Dj)                                   #ni x nj
            self.wjk = self.wjk + _rate * (delta_jk + _lambda * self.wjk)
            self.wij = self.wij + _rate * (delta_ij + _lambda * self.wij)
            """

        if n == _epoch:
            print "Warning: maximum number of iteration (%g) reached, norm=%g"%(_epoch, err)


def test_regression():
    p = 10
    n = 30
    X = np.random.multivariate_normal(np.zeros(p),np.eye(p),(n,))
    X = np.concatenate((X,np.ones(n).reshape(n,1)),axis=1)
    beta = np.random.randn(p+1).reshape(p+1,1)
    y = X.dot(beta)

    _in = LinearLayer(p)
    hidden = SigmoidLayer(6)
    output = LinearLayer(1)
    nn = NeuralNetWork(_in, hidden, output, bias=True)
    nn.train(X[:,:-1],y)
    pred = nn.predict(X)

    mae = np.mean(np.abs(y-pred))
    rmse = np.sqrt(np.mean(np.power(y-pred,2)))
    print "mae=%g, rmse=%g"%(mae,rmse)

