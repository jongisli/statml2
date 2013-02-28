#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from math import log

img_format = 'png'

def get_data(datafile):
    f = open(datafile)
    data = np.loadtxt(f)
    f.close()
    return data

def split_iris_data(datafile):
    data = get_data(datafile)
    class_0 = data[data[:,2] == 0]
    class_1 = data[data[:,2] == 1]
    class_2 = data[data[:,2] == 2]

    return (class_0, class_1, class_2)

def plot_classes(datafile):
    setosa, virginica, versicolor = split_iris_data(datafile)
    
    plt.scatter(setosa[:,0], setosa[:,1],
                color='red', label='Iris setosa')
    plt.scatter(virginica[:,0], virginica[:,1],
                color='green', label='Iris virginica')
    plt.scatter(versicolor[:,0], versicolor[:,1],
                color='blue', label='Iris versicolor')

    data = get_data(datafile) 
    plt.show()
    plt.close()

def k_closest(P,k):
  def model(x):
    Map = lambda(y):[np.linalg.norm(x-y[0:2]),y]
    l_ = map(Map, P)
    l_.sort(key=lambda(x): x[0])
    ks = map(lambda(x) : x[1],l_)[0:k]
    
    cum = np.array([0,0,0])
    for x in ks:
        cum[x[2]] = cum[x[2]]+1
    
    return np.argmax(cum)
  return model


def linear_discriminant_analysis_estimates(datafile):
    data = split_iris_data(datafile)
    
    l = float(sum([klass[:,0].size for klass in data]))
    m = len(data)

    Pl_k = []
    Mu_k = []
    Cov = np.zeros((2,2))
    for klass in data:
        l_k = float(klass[:,0].size)
        Pl_k.append(l_k/l)
        
        mu_k = np.array((1/l_k) * klass.sum(axis=0)[:-1])
        Mu_k.append(mu_k)
        
        Cov += (1/(l-m))*sum([np.matrix(x[:-1] - mu_k).T * np.matrix(x[:-1] - mu_k) for x in klass])

    return (Pl_k, Mu_k, Cov)

def discrimination_function(Pl, Mu, Cov):
    def delta(x):
        CovI =  np.linalg.inv(Cov)
        return x.dot(CovI).dot(Mu.T) - 0.5*Mu.dot(CovI).dot(Mu.T) + log(Pl)
    return delta

def decicion_function(data_train):
    Pl_k, Mu_k, Cov = linear_discriminant_analysis_estimates(data_train)
    discr_funcs = [discrimination_function(Pl, Mu, Cov) for Pl,Mu in zip(Pl_k,Mu_k)]
    def y(x):
        delta = [d_func(x) for d_func in discr_funcs]
        return delta.index(max(delta))
    return y

def model_error(data_test, model):
    y = model

    f = open(data_test)
    data = np.loadtxt(f)
    f.close()

    Y = [y(np.array([a,b])) for a,b,_ in data]
    T = data.take([2], axis=1)
    Eq = np.equal(Y,T.T)
    
    return Eq[Eq].size / float(Eq.size)
        
if __name__ == "__main__":
    P = get_data('data/irisTrain.dt')
    
    for k in [1,3,7,9,11,13,15,31]:
        print 'error for %d is %f' % (k,model_error('data/irisTest.dt',k_closest(P,k)))

