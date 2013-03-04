#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import split_data as splitter

img_format = 'png'

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

def plot_alpha_error(train_file, test_file):
    alpha_range = np.arange(0.001, 0.5, 0.01)
    errors1 = []
    errors2 = []
    for alpha in alpha_range:
        bayes_model1 = bayesian_model(train_file, [3, 6, 7, 8], [1], alpha, 1)
        bayes_model2 = bayesian_model(train_file, [7], [1], alpha, 1)
        errors1.append(root_mean_square_error(test_file, [3, 6, 7, 8], [1], bayes_model1))
        errors2.append(root_mean_square_error(test_file, [7], [1], bayes_model2))

    ml_model1 = maximum_likelihood_model(train_file, [3, 6, 7, 8], [1])
    ml_model2 = maximum_likelihood_model(train_file, [7], [1])
    
    ml_error1 = root_mean_square_error(test_file, [3, 6, 7, 8], [1], ml_model1)
    ml_error2 = root_mean_square_error(test_file, [7], [1], ml_model2)

    
    plt.plot(alpha_range, errors1, label='$MAP_1$')
    plt.plot(alpha_range, [ml_error1]*len(alpha_range), label='$ML_1$')
    plt.legend()
    plt.savefig('images/rms_selection1.%s' % img_format, format=img_format)
    plt.close()
    
    plt.plot(alpha_range, errors2, label='$MAP_2$')
    plt.plot(alpha_range, [ml_error2]*len(alpha_range), label='$ML_2$')
    plt.legend()
    plt.savefig('images/rms_selection2.%s' % img_format, format=img_format)
    plt.close()

    
def average_error(design_columns, target_columns, N):
    error = 0
    for i in range(0,N):
        splitter.create_training_and_test_sets('data/bodyfatdata.txt', training_ratio = 0.8)
        model = maximum_likelihood_model('data/bodyfat_training.txt',
                                         design_columns, target_columns)
        error += root_mean_square_error('data/bodyfat_test.txt',
                                        design_columns, target_columns, model)
    return error / float(N)

    
if __name__ == "__main__":
    #print "The average RMS error for selection 1:"
    #print average_error([3, 6, 7, 8], [1], 100)
    #print "The average RMS error for selection 2:"
    #print average_error([7], [1], 100)

    #splitter.create_training_and_test_sets('data/bodyfatdata.txt', training_ratio = 0.8)
    print "Plotting MAP vs ML ..."
    plot_alpha_error('data/bodyfat_training.txt', 'data/bodyfat_test.txt')
