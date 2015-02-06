'''
#=============================================================================
#     FileName: functions.py
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2015-02-05 16:29:56
#   LastChange: 2015-02-05 16:29:56
#      History:
#=============================================================================
'''
import numpy as np

f_linear      = lambda x: x
fprime_linear = lambda x: np.ones(x.shape)

def my_exp(x, _min=-100, _max=100):
    return np.exp(np.clip(x, _min, _max))

f_exp      = lambda x: my_exp(x)
fprime_exp = lambda x: my_exp(x)


def f_sigmoid(x):
    return 1. / (1 + my_exp(-x))

def fprime_sigmoid(x):
    f = f_sigmoid(x)
    return f * (1 - f)

'''
sinh(x) = (e^x - e^-x) / 2
cosh(x) = (e^x + e^-x) / 2
tanh(x) = sinh(x) / cosh(x)
'''
f_tanh = lambda x: np.tanh(x)
fprime_tanh = lambda x: 1 - np.power(f_tanh(x),2)

