import numpy as np

def calc_l1_norm(t,z):
    return np.sum(np.abs(t-z))

def reg_error(t,z):
    return np.sum(np.abs(t-z))

def reg_mae(t,z):
    return np.mean(np.abs(t-z))

def reg_rmse(t,z):
    return np.sqrt(np.mean(np.power(t-z,2)))

def reg_r2(t,z):
    return np.power(np.corrcoef(t.flatten(),z.flatten())[0][1],2)

def class_error(t,z):
    err = 0
    for i in xrange(t.shape[0]):
        j = z[i].argmax()
        if t[i][j] == 0:
            err += 1
    return err

def cross_validation(X,y):
    pass

