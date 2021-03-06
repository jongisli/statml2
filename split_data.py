#!/usr/bin/python
import numpy as np

"""
This function splits the bodyfat training file into
training and test files according to the training_ratio
parameter
"""
def create_training_and_test_sets(datafile, training_ratio = 0.2):
    f = open(datafile)
    data = np.loadtxt(f)
    n,_ = data.shape

    # We start by randomly permuting the data
    # so we can then directly split up the array
    # into training and test array.
    permuted_data = np.random.permutation(data)
    training_data = permuted_data[:int(n*training_ratio)]
    test_data = permuted_data[int(n*training_ratio):]

    np.savetxt('data/bodyfat_training.txt', training_data)
    np.savetxt('data/bodyfat_test.txt', test_data)
        
    return (training_data,test_data)

if __name__ == "__main__":
    training, test = create_training_and_test_sets('data/bodyfatdata.txt')
    print str(training[0]) + " and " + str(training[-1])
    print str(test[0]) + " and " + str(test[-1])
    print training.shape
    print test.shape
