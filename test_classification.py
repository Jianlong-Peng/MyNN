import numpy as np
import matplotlib.pyplot as plt
from layer import *
from nn import NeuralNetwork,convert_y
from validation import class_error
import sys

def read_iris():
    inf = open("iris.tab",'r')
    line = inf.readline()
    X = []
    y = []
    for line in inf:
        line = line.split()
        X.append(map(float,line[1:-1]))
        y.append(line[-1])
    inf.close()
    return np.asarray(X),np.asarray(y)

def test_classification(verbose):
    X,y = read_iris()
    y2,unique_label = convert_y(y)
    _in = LinearLayer(X.shape[1])
    hidden = SigmoidLayer(6)
    output = SigmoidLayer(y2.shape[1])
    model = NeuralNetwork(_in,hidden,output,bias=True)
    print "number of parameters:",model.num_parameters()
    print "number of samples:",X.shape[0]
    print "number of independent variables:",X.shape[1]
    print "to train the neural network..."
    model.train(X,y2,_epsilon=X.shape[0]/20.,verbose=verbose,estimate=class_error)

    pred = model.predict(X)
    error = class_error(y2,pred)
    correct = X.shape[0] - error
    print "correct=%d, error=%d, total=%d"%(correct,error,X.shape[0])


if __name__ == "__main__":
    argv = sys.argv
    verbose = False
    if len(argv)==2 and argv[1]=="--verbose":
        verbose = True
    test_classification(verbose)
