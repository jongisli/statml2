#!/usr/bin/python
import numpy as np

def design_matrix(data, columns):
    selection = data.take(columns, axis=1)
    n,_ = selection.shape
    ones = np.ones([n,1])
    return np.hstack((ones, selection))


def maximum_likelihood_model(datafile, design_columns, target_columns):
    f = open(datafile)
    data = np.loadtxt(f)
    f.close()

    t = data.take(target_columns, axis=1)
    Phi = design_matrix(data, design_columns)
    w = np.linalg.pinv(Phi).dot(t)
    w = w.T[0]
    w_0 = w[0]
    w_rest = w[1:]

    def y(x):
        return w_0 + w_rest.dot(x.T)

    return y


def root_mean_square_error(datafile, design_columns, target_columns, model):
    f = open(datafile)
    data = np.loadtxt(f)
    f.close()
    
    t = data.take(target_columns, axis=1)
    X = data.take(design_columns, axis=1)
    N,_ = t.shape

    RMS = ((1/float(N))*sum([(t_n - model(x_n))**2 for t_n,x_n in zip(t,X)]))**0.5

    return RMS[0]
    
if __name__ == "__main__":
    model1 = maximum_likelihood_model('data/bodyfat_training.txt',
                                    [3, 6, 7, 8], [1])
    model2 = maximum_likelihood_model('data/bodyfat_training.txt',
                                    [7], [1])

    print root_mean_square_error('data/bodyfat_test.txt',
                                    [3, 6, 7, 8], [1], model1)
    print root_mean_square_error('data/bodyfat_test.txt',
                                    [7], [1], model2)
