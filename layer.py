'''
#=============================================================================
#     FileName: layer.py
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2015-02-06 10:47:04
#   LastChange: 2015-02-06 11:05:14
#      History:
#=============================================================================
'''
import functions

__all__ = ['Layer', 'LinearLayer', 'SigmoidLayer', 'TanhLayer']

class Layer:
    def __init__(self, n, f, fprime):
        self.n = n
        self.f = f
        self.fprime = fprime

    def activate(self, _input, weight):
        '''
        Parameter
        =========
        _input: np.ndarray with shape (m, ni)
        weight: np.ndarray with shape (ni, nj)

        Return
        ======
        np.ndarray with shape (m, nj)
        '''
        assert _input.shape[1]==weight.shape[0] and weight.shape[1]==self.n
        net = _input.dot(weight)
        _output = self.f(net)
        return _output


class LinearLayer(Layer):
    def __init__(self, n):
        Layer.__init__(self, n, functions.f_linear, functions.fprime_linear)

class SigmoidLayer(Layer):
    def __init__(self, n):
        Layer.__init__(self, n, functions.f_sigmoid, functions.fprime_sigmoid)

class TanhLayer(Layer):
    def __init__(self, n):
        Layer.__init__(self, n, functions.f_tanh, functions.fprime_tanh)


