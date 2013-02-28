#!/usr/bin/python
import numpy as np
#import matplotlib.pyplot as plt
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
    _, axes = plt.subplots()
    
    plt.scatter(setosa[:,0], setosa[:,1],
                color='red', label='Iris setosa')
    plt.scatter(virginica[:,0], virginica[:,1],
                color='green', label='Iris virginica')
    plt.scatter(versicolor[:,0], versicolor[:,1],
                color='blue', label='Iris versicolor')
    circ = plt.Circle((0,0), radius=6.00, fill=False, color='g')
    axes.add_patch(circ)
    circ = plt.Circle((0,0), radius=5.50, fill=False, color='b')
    axes.add_patch(circ)

    data = get_data(datafile)

    rad,circ = getCircle(data,[6.0,0.27],3)
    axes.add_patch(circ)
    print rad

    plt.show()
    plt.close()

def k_closest(P,x,k):
    Map = lambda(y):[np.linalg.norm(x-y[0:2]),y]
    l_ = map(Map, P)
    l_.sort(key=lambda(x): x[0])
    ks = map(lambda(x) : x[1],l_)[0:k]
    
    cum = np.array([0,0,0])
    for x in ks:
        cum[x[2]] = cum[x[2]]+1
    
    return np.argmax(cum)

def lenFilterMap(Pred,Map,l):
    c = 0
    for e in l:
        if Pred(Map(e)):
	    c = c+1
    return c

def sphere_count(P,x,rad):
    Map = lambda(y):np.linalg.norm(x-y)
    Pred = lambda(d) : d <= rad
    return lenFilterMap(Pred,Map,P)

def sphere_rad(P,x,k,Rmin,Rmax):
    eps = 0.0005
    i,j=Rmin,Rmax
    while(j-i > eps):
       #| too small | unknown | too large
       #Rmin         i         j         Rmax
       #
       #The radius of the circle lies inside [i,j)
       
       h=(j-i)/2.0
       
       c=(sphere_count(P,x,h))
       if c < k:
           i = h
       elif c > k:
           j = h
       else:
           return h
    return h

def getCircle(P,x,k):
    P=P[:,0:2]
    norms = map(lambda(x): np.linalg.norm(x),P)
    Rmin = min(norms)
    Rmax = max(norms)

    rad=sphere_rad(P,x,k,Rmin,Rmax)
   
    return rad,plt.Circle(x, radius=rad, fill = False)

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
        
if __name__ == "__main__":
    P = get_data('data/irisTrain.dt')

    print k_closest(P,[5.0,0.35],3)
    #dfunc =  decicion_function('data/irisTrain.dt')
    #print dfunc(np.array([5.5, 0.25]))
