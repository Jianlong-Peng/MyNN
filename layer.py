'''
#=============================================================================
#     FileName: layer.py
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2015-02-06 10:47:04
#   LastChange: 2015-02-15 23:10:58
#      History:
#=============================================================================
'''
import functions
import numpy as np

__all__ = ['Layer', 'LinearLayer', 'SigmoidLayer', 'TanhLayer']

class Layer:
    def __init__(self, n, f, fprime):
        self.n = n
        self.f = f
        self.fprime = fprime

    def activate(self, _input, weight, bias=None):
        '''
        Parameter
        =========
        _input: np.ndarray with shape (m, ni)
        weight: np.ndarray with shape (ni, nj)
        bias  : np.ndarray with shape (1, nj) or None

        Return
        ======
        np.ndarray with shape (m, nj)
        '''
        assert weight is not None
        assert _input.shape[1] == weight.shape[0]
        if bias is not None:
            assert weight.shape[1] == bias.shape[1]
        m = _input.shape[0]
        _output = _input.dot(weight)
        if bias is not None:
            _output += np.ones(m).reshape(m,1).dot(bias)
        return self.f(_output)


class LinearLayer(Layer):
    def __init__(self, n):
        Layer.__init__(self, n, functions.f_linear, functions.fprime_linear)

class SigmoidLayer(Layer):
    def __init__(self, n):
        Layer.__init__(self, n, functions.f_sigmoid, functions.fprime_sigmoid)

class TanhLayer(Layer):
    def __init__(self, n):
        Layer.__init__(self, n, functions.f_tanh, functions.fprime_tanh)


