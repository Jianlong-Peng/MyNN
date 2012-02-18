from random import random
import numpy as np
from layer import Layer
from validation import calc_l1_norm

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
         = \sum_k{D_k * w_jk} * f'(net_j) * x_i + lambda*w_ij
  let D_j = \sum_k{D_k * w_jk}*f'(net_j) ............... (4)
  then, \partial{J}{w_ij} = D_j*x_i + lambda*w_ij.......... (5)
  therefore, the update rule of w_ij could be:
  w^{t+1}_ij = w^{t}_ij - rate * (D_j*x_i + lambda*w_ij)... (6)



More general cases (L-layer NN)
1. feed-forward part
   in layer `l` (l = 1 to L), 
   NET(l) = X(l-1) .dot(B(l-1,l)) + I.dot(B(l)),  where B(l) is bias
   Y(l)   = f(NET(l))
   X(l)   = Y(l)

2. back-propgation
   1) for the output layer L
      D(L) = -(T-Z) * f'(NET(L))
      \partial{J}{B(L-1,L)} = X(L-1).T.dot(D(L)) + lambda * B(L-1,L)
      \partial{J}{B(L)}     = I.T.dot(D(L)) + lambda * B(L)
   2) for layer (l-1) to l
      D(l) = D(l+1).dot(B(l,l+1).T) * f'(NET(l))
      \partial{J}{B(l-1,l)} = X(l-1).T.dot(D(l)) + lambda * B(l-1,l)
      \partial{J}{B(l)}     = I.T.dot(D(l)) + lambda * B(l)
"""



#Attention:
# 1. how to deal with NN with more than one output neuro???
class NeuralNetwork:
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
        self.weight = [None for i in xrange(len(self.layers))]
        self.bias = [None for i in xrange(len(self.layers))]
        for i in xrange(len(self.layers)-1):
            if kargs.get("bias",False):
                self.bias[i+1] = np.random.random(self.layers[i+1].n).reshape(1,self.layers[i+1].n)
            self.weight[i+1] = \
                    np.asarray([[np.random.random() for j in xrange(self.layers[i].n)] for k in xrange(self.layers[i+1].n)]).T
            #print self.weight[i+1].shape

    def num_parameters(self):
        n = 0
        for item in self.weight[1:]:
            n += (item.shape[0] * item.shape[1])
        for item in self.bias:
            if item is not None:
                n += (item.shape[0] * item.shape[1])
        return n

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
        Y = np.copy(X)
        for i in xrange(1,len(self.layers)):
            Y = self.layers[i].activate(Y, self.weight[i], self.bias[i])
        return Y

    def train(self, X, y, _epoch=1E5, _epsilon=0.01, _rate=1E-3, _lambda=0, verbose=False, estimate=calc_l1_norm):
        '''
        Parameter
        =========
        X: np.ndarray with shape (m, ni)
           where, m is number of samples, ni is number of predictors
        y: np.ndarray with shape (m,) or (m,1) or (m,p)
        _epoch:   maximum number of iterations
        _epsilon: threshold, stop when `_epsilon < estimate(t,z)`
        _rate:    learning rate
        _lambda:  parameter for L2-norm
        verbose:  bool
        estimate: function to estimate training
                  `float f(target,predict)`

        '''
        X = np.copy(X)
        if y.ndim == 1:
            t = y.reshape(y.shape[0],1)
            assert self.layers[-1].n == 1
        else:
            t = np.copy(y)
            assert y.shape[1] == self.layers[-1].n

        m = X.shape[0]
        one_col = np.ones(m).reshape(m,1)
        n = 0
        err = 1E6
        while n < _epoch:
            n += 1
            if verbose:
                print "iter #%d, norm=%g"%(n,err)
            #1. forward
            net = [None for _ in xrange(len(self.layers))]
            net[0] = X
            z = [None for _ in xrange(len(self.layers))]
            z[0] = X
            for i in xrange(1,len(self.layers)):
                net[i] = z[i-1].dot(self.weight[i])
                if self.bias[i] is not None:
                    #print net[i].shape,one_col.shape,self.bias[i].shape
                    net[i] += one_col.dot(self.bias[i])
                z[i] = self.layers[i].f(net[i])
            #2. estimate
            err = estimate(t,z[-1])
            if err < _epsilon:
                break
            #3. backpropgation
            #error in the output layer
            D = (t-z[-1])*self.layers[-1].fprime(net[-1])
            delta3 = z[-2].T.dot(D)
            if self.bias[-1] is not None:
                delta4 = one_col.T.dot(D)
            else:
                delta4 = None
            for i in xrange(len(net)-2, 0, -1):
                #error backpropgated from the last layer
                D = D.dot(self.weight[i+1].T) * self.layers[i].fprime(net[i])
                delta1 = z[i-1].T.dot(D)
                if self.bias[i+1] is not None:
                    delta2 = one_col.T.dot(D)
                else:
                    delta2 = None
                #update weight between the current and last layer
                self.weight[i+1] = self.weight[i+1] + _rate * (delta3 + _lambda * self.weight[i+1])
                if self.bias[i+1] is not None:
                    self.bias[i+1] = self.bias[i+1] + _rate * (delta4 + _lambda * self.bias[i+1])
                delta3 = delta1
                delta4 = delta2
            self.weight[1] = self.weight[1] + _rate * (delta3 + _lambda * self.weight[1])
            if self.bias[1] is not None:
                self.bias[1] = self.bias[1] + _rate * (delta4 + _lambda * self.bias[1])
        
        #print "#iter:",n
        #print "error:",err
        if n == _epoch:
            print "Warning: maximum number of iteration (%g) reached, norm=%g"%(_epoch, err)


def convert_y(y):
    '''
    OBJ
    ===
    for classification problem, to convert class label
    to those suitable for neural network

    Parameter
    =========
    y: np.ndarray with shape (m,) or (m,1) or (1,m)

    Return
    ======
    (new_y,unique_label)
    each is np.ndarray with shape (m,p) and (p,)
    '''
    assert len(y.shape)==1 or (len(y.shape)==2 and (y.shape[0]==1 or y.shape[1]==1))
    y = y.flatten()
    labels = np.unique(y).tolist()
    labels.sort()
    m = y.shape[0]
    p = len(labels)
    new_y = np.zeros(m*p).reshape(m,p)
    for i in xrange(m):
        j = labels.index(y[i])
        new_y[i][j] = 1
    return new_y,np.asarray(labels)

