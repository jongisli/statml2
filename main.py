#!/usr/bin/python
import classification as cla
import regression as reg
import split_data as spl
import numpy as np

if (raw_input("II.1 Split the bodyfat dataset into training and test sets? (y/n):") == "y"):
    spl.create_training_and_test_sets('data/bodyfatdata.txt')
    print "... Saved datasets as data/bodyfat_training.txt and data/bodyfat_test.txt"
    print


if (raw_input("II1.1 Compute the RMS for the maximum likelihood solution? (y/n):") == "y"):
    ml_model1 = reg.maximum_likelihood_model('data/bodyfat_training.txt',
                                             [3, 6, 7, 8], [1])
    ml_model2 = reg.maximum_likelihood_model('data/bodyfat_training.txt', [7], [1])
    ml_error1 = reg.root_mean_square_error('data/bodyfat_test.txt',
                                       [3, 6, 7, 8], [1], ml_model1)
    ml_error2 = reg.root_mean_square_error('data/bodyfat_test.txt', [7], [1], ml_model2)
    print "The RMS error of the test set for selection 1:"
    print ml_error1
    print "The RMS error of the test set for selection 2:"
    print ml_error2
    print


if (raw_input("II1.2 Compute the RMS for the maximum a posteriori solution? (y/n):") == "y"):
    reg.plot_alpha_error('data/bodyfat_training.txt','data/bodyfat_test.txt')
    print "Saved plots as images/rms_selection1.png and images/rms_selection2.png"
    print

    
if (raw_input("II2.1 Scatter plot of Iris training data and training and test errors of the LDA? (y/n):") == "y"):
    cla.plot_classes('data/irisTrain.dt')
    print "Saved scatter plot as images/iris_scatter.png"
    print

    model = cla.decicion_function('data/irisTrain.dt')
    print "The training error of LDA:"
    print cla.model_error('data/irisTrain.dt', model)
    print "The test error of LDA:"
    print cla.model_error('data/irisTest.dt', model)
    print

if (raw_input("II2.3 Training and test error of k-NN? No will skip II2.4 (y/n):") == "y"):
    P = cla.get_data('data/irisTrain.dt')

    print 'Error with norm as a metric on irisTrain:\\\\' 
    NormTrain,NormPerClassTrain = cla.NNerrorTable('data/irisTrain.dt',cla.k_closest_norm,P)
    print 'Error with norm as a metric on irisTest:\\\\' 
    NormTest,NormPerClassTest = cla.NNerrorTable('data/irisTest.dt',cla.k_closest_norm,P)

    if (raw_input("II2.4 Training and test error of k-NN with d metric? (y/n):") == "y"):

        print 'Error with d as a metric on irisTrain:\\\\'
        MTrain,MPerClassTrain = cla.NNerrorTable('data/irisTrain.dt',cla.k_closest_M,P)

        print 'Error with d as a metric on irisTest:\\\\'
        MTest,MPerClassTest = cla.NNerrorTable('data/irisTest.dt',cla.k_closest_M,P)
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

if (raw_input("II2.5 Training and test error of LDA with d metric? (y/n):") == "y"):
    M = np.array([[1,0],[0,10]])
    datafiles = ['data/irisTrain.dt', 'data/irisTest.dt']
    cla.scale_data(datafiles, M)

    modelScaled = cla.decicion_function('data/irisTrain.dt.scaled')
    print "The training error of LDA (scaled):"
    print cla.model_error('data/irisTrain.dt.scaled', modelScaled)
    print "The test error of LDA (scaled):"
    print cla.model_error('data/irisTest.dt.scaled', modelScaled)
    print
