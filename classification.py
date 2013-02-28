#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

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

def FilterMap(Pred,Map,l):
    return [x if Pred(Map(x)) for x in l]

def k_closest(l,k):
    f = F

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

if __name__ == "__main__":
    print split_iris_data('data/irisTrain.dt')
    plot_classes('data/irisTrain.dt')
    
