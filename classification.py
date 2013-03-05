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
#     neigbours
#Ret: a function that takes a point as input
#     and gives it's likeliest class based
#     on k-NN with training set P, using
#     the given metric to measure distances
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

#Pre: P is a set of points
#     k is number of
#     neigbours
#Ret: a function that takes a point as input
#     and gives it's likeliest class based
#     on k-NN with training set P, using
#     the euclidean norm to measure distances
def k_closest_norm(P,k):
    return k_closest(norm,P,k)

#Pre: P is a set of points
#     k is number of
#     neigbours
#Ret: a function that takes a point as input
#     and gives it's likeliest class based
#     on k-NN with training set P, using
#     d(x,y)=||Mx-My|| with M={{1,0},{0,10}}
#     to measure distances.
def k_closest_M(P,k):
    M=np.array([[1,0],[0,10]])
    return k_closest(lambda(x,y) : norm(M.dot(x) - M.dot(y)) ,P,k)

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
    
    return 1 - Eq[Eq == True].size / float(Eq.size)

def model_error_class(data_test, model):
    y = model

    f = open(data_test)
    data = np.loadtxt(f)
    f.close()

    Error = [None]*3
    for c in range(0,3):
       class0=data[data[:,2]==c]
       Y = [y(np.array([a,b])) for a,b,_ in class0]
       T = class0.take([2], axis=1)
       Eq = np.equal(Y,T.T)
       Error[c]=1 - Eq[Eq == True].size / float(Eq.size)
    
    return Error

def scale_data(datafiles, M):
    for datafile in datafiles:
        data = np.loadtxt(datafile)
        data[:,[0,1]] = data.take([0,1], axis=1).dot(M)
        np.savetxt(datafile + '.scaled', data, fmt='%.3e',)

def NNerrorTable(datapath,model,P):
    print '\\begin{center}'
    print '\\begin{tabular}{l|l}'
    print ' k & total error & class 0 & class 1 & class 2\\\\'
    Error = [None]*32
    ErrorPerClass = [None]*32
    for k in [1,3,5,7]:
        Error[k]=model_error(datapath,model(P,k))
        ErrorPerClass[k]=model_error_class(datapath,model(P,k))
        print '%2d & %f & %f & %f & %f\\\\' % (k, Error[k],ErrorPerClass[k][0],ErrorPerClass[k][1],ErrorPerClass[k][2])
    print '\\end{tabular}'
    print '\\end{center}'
    return Error,ErrorPerClass

if __name__ == "__main__":

    M = np.array([[1,0],[0,10]])
    datafiles = ['data/irisTrain.dt', 'data/irisTest.dt']
    scale_data(datafiles, M)
    
    model = decicion_function('data/irisTrain.dt')
    print "The training error of LDA:"
    print model_error('data/irisTrain.dt', model)
    print "The test error of LDA:"
    print model_error('data/irisTest.dt', model)
    print
    
    print "SCALED DATA:"
    modelScaled = decicion_function('data/irisTrain.dt.scaled')
    print "The training error of LDA (scaled):"
    print model_error('data/irisTrain.dt.scaled', modelScaled)
    print "The test error of LDA (scaled):"
    print model_error('data/irisTest.dt.scaled', modelScaled)
    print
    
    P = get_data('data/irisTrain.dt')
    print 'Error with norm as a metric on irisTrain:\\\\' 
    NormTrain,NormPerClassTrain = NNerrorTable('data/irisTrain.dt',k_closest_norm,P)

    print 'Error with norm as a metric on irisTest:\\\\' 
    NormTest,NormPerClassTest = NNerrorTable('data/irisTest.dt',k_closest_norm,P)
    print 'Error with d as a metric on irisTrain:\\\\'
    MTrain,MPerClassTrain = NNerrorTable('data/irisTrain.dt',k_closest_M,P)

    print 'Error with d as a metric on irisTest:\\\\'
    MTest,MPerClassTest = NNerrorTable('data/irisTest.dt',k_closest_M,P)

    print 'Difference in error between Norm and d on irisTrain:\\\\'
    print '\\begin{tabular}{ll}'
    print ' k & total error & class 0 & class 1 & class 2\\\\'
    for k in [1,3,5,7]:
        print '%2d & %f' % (k,NormTrain[k]-MTrain[k]),
	for c in [0,1,2]:
	    print '& %f' % (NormPerClassTrain[k][c]-MPerClassTrain[k][c]),
	print '\\\\'
    print '\\end{tabular}'
    print 'Difference in error between Norm and d on irisTest:\\\\'
    print '\\begin{center}'
    print '\\begin{tabular}{l|l}'
    print ' k & total error & class 0 & class 1 & class 2\\\\'
    for k in [1,3,5,7]:
        print '%2d & %f' % (k,NormTest[k]-MTest[k]),
	for c in [0,1,2]:
	    print '& %f' % (NormPerClassTest[k][c]-MPerClassTest[k][c]),
	print '\\\\'
    print '\\end{tabular}'
    print '\\end{center}'
