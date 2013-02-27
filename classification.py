#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

def split_iris_data(datafile):
    f = open(datafile)
    data = np.loadtxt(f)
    f.close()

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
    plt.show()
    plt.close()


if __name__ == "__main__":
    print split_iris_data('data/irisTrain.dt')
    
