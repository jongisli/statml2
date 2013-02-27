#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import split_data as splitter

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

def bayesian_model(datafile, design_columns, target_columns, alpha, beta):
    f = open(datafile)
    data = np.loadtxt(f)
    f.close()

    t = data.take(target_columns, axis=1)
    Phi = design_matrix(data, design_columns)

    M = len(design_columns) + 1
    S_0 = (1/alpha)*np.identity(M)
    S_0I = np.linalg.inv(S_0)
    m_0 = np.zeros((M,1))

    S_n = np.linalg.inv(S_0I + beta*(Phi.T).dot(Phi))
    m_n = S_n.dot(S_0I.dot(m_0) + beta*Phi.T.dot(t))
    w = m_n
    
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

def plot_alpha_error(train_file, test_file, design_columns, target_columns):
    alpha_range = np.arange(0.001, 0.5, 0.01)
    errors = []
    for alpha in alpha_range:
        bayes_model = bayesian_model(train_file, design_columns, target_columns, alpha, 1)
        errors.append(root_mean_square_error(test_file, design_columns, target_columns, bayes_model))

    ml_model = maximum_likelihood_model(train_file, design_columns, target_columns)
    ml_error = root_mean_square_error(test_file, design_columns, target_columns, ml_model)

    plt.plot(alpha_range, errors)
    plt.plot(alpha_range, [ml_error]*len(alpha_range))
    plt.show()
    plt.close()

    
if __name__ == "__main__":
    #splitter.create_training_and_test_sets('data/bodyfatdata.txt', training_ratio = 0.8)

    plot_alpha_error('data/bodyfat_training.txt', 'data/bodyfat_test.txt', [3, 6, 7, 8], [1])
    plot_alpha_error('data/bodyfat_training.txt', 'data/bodyfat_test.txt', [7], [1])
