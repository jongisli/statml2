#!/usr/bin/python
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from math import log

img_format = 'png'

#Pre: datafile points to an
#     existant file with no
#     restrictions.
#Ret: The data inside the
#     datafile
def get_data(datafile):
    f = open(datafile)
    data = np.loadtxt(f)
    f.close()
    return data

#Pre: datafile is an existant
#     iris data file
#Ret: Names of the different classess
def split_iris_data(datafile):
    data = get_data(datafile)
    class_0 = data[data[:,2] == 0]
    class_1 = data[data[:,2] == 1]
    class_2 = data[data[:,2] == 2]

    return (class_0, class_1, class_2)

#Pre: datafile is an existant
#     iris data file
#Post: shows a scatterplot of
#      data in datafile
def plot_classes(datafile):
    setosa, virginica, versicolor = split_iris_data(datafile)
    
    plt.scatter(setosa[:,0], setosa[:,1],
                color='red', label='Iris setosa')
    plt.scatter(virginica[:,0], virginica[:,1],
                color='green', label='Iris virginica')
    plt.scatter(versicolor[:,0], versicolor[:,1],
                color='blue', label='Iris versicolor')
    plt.xlabel('Length (mm)')
    plt.ylabel('Width (cm)')
    plt.legend()
    plt.savefig('images/iris_scatter.%s' % img_format, format=img_format)
    plt.close()

#Pre: metric is a metric
#     P is a set of points
#     k is number of
#     neigphourss
def k_closest(metric,P,k):
  def model(x):
    Map = lambda(y):[metric(x-y[0:2]),y]
    l_ = map(Map, P)
    l_.sort(key=lambda(x): x[0])
    ks = map(lambda(x) : x[1],l_)[0:k]
    
    cum = np.array([0,0,0])
    for x in ks:
        cum[x[2]] = cum[x[2]]+1
    
    return np.argmax(cum)
  return model

def k_closest_norm(P,k):
    return k_closest(norm,P,k)

def k_closest_M(P,k):
    M=np.array([[1,0],[0,10]])
    return k_closest(lambda(x,y) : norm(M.dot(x) - M.dot(y)) ,P,k)

def linear_discriminant_analysis_estimates(datafile,xScaler=lambda(x):x):
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
    
    return 1 - Eq[Eq].size / float(Eq.size)
        
if __name__ == "__main__":
    model = decicion_function('data/irisTrain.dt')

    print "The training error of LDA:"
    print model_error('data/irisTrain.dt', model)
    print "The test error of LDA:"
    print model_error('data/irisTest.dt', model)
    
    """
    P = get_data('data/irisTrain.dt')
    
    #M=np.array([[1,0],[0,10]])
    #linear_discriminant_analysis_estimates(P,xScaler=lambda(x,y,c):M.dot(x))
    
    for k in [1,3,7,9,11,13,15,31]:
        print 'norm error for %d is %f' % (k,model_error('data/irisTest.dt',
	    k_closest_norm(P,k)))
    for k in [1,3,7,9,11,13,15,31]:
        print 'M error for %d is %f' % (k,model_error('data/irisTest.dt',
	    k_closest_M(P,k)))
    """
