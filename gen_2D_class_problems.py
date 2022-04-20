import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

"""Script to generate various 2-dimensional K-classification problems 
to test novel training schemes on."""

problem_type = "spiral"  # choose problem type to generate 
                        # until now, allowed is "trigo" for a trigonometric data set
                        # and "spiral" for Swiss Roll type spirals.
                        # details of problems need to be specified below.
                        

if problem_type == "trigo":
    
    a = 1    # constants
    c = 0.02
    b = 10   # larger = more difficult (frequency of trigonometric curves)
    D = 2    # dimensionality
    K = 2    # number of classes
    Ntrain = 500   # number of points per class in train set
    Ntest = 100    # number of points per class in test set    
    
    
    ## Create Train Set
    N = Ntrain
    X = np.zeros((N*K,D))                   # feature matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8')        # class labels
    
    for j in range(K):
        t = rnd.uniform(size=N)
        rand = rnd.normal(size=N)
        x1 = (a*t).reshape(N,1)
        x2 = (np.cos( (-1)**j * b*t*np.pi + j*np.pi/2) + c*rand ).reshape(N,1)
        X[N*j:N*(j+1)] = np.concatenate((x1,x2), axis=1)
        y[N*j:N*(j+1)] = j
    

    np.save("trig_train_features.npy", X)   # write train features and labels
    np.save("trig_train_labels.npy", y)
    
    ## Create Test Set
    N = Ntest 
    X = np.zeros((N*K,D))                   # feature matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8')        # class labels
    
    for j in range(K):
        t = rnd.uniform(size=N)
        rand = rnd.normal(size=N)
        x1 = (a*t).reshape(N,1)
        x2 = (np.cos( (-1)**j * b*t*np.pi + j*np.pi/2) + c*rand ).reshape(N,1)
        X[N*j:N*(j+1)] = np.concatenate((x1,x2), axis=1)
        y[N*j:N*(j+1)] = j

    np.save("trig_test_features.npy", X)    # write test features and labels
    np.save("trig_test_labels.npy", y)



elif problem_type == "spiral":
    a = 2       # constants
    p = 1
    c = 0.02
    b = 4       # larger = more difficult (number of spiral turns)
    D = 2       # dimensionality
    K = 2       # number of classes
    Ntrain = 1   # number of points per class in train set
    Ntest = 100    # number of points per class in test set
    
    ## Create Train Set                                   
    N = Ntrain
    X = np.zeros((N*K,D))                       # feature matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8')            # class labels
   
    for j in range(K):
        t = rnd.uniform(size=N)
        rand = rnd.normal(size=N)
        x1 = (a*t**p * np.sin( 2*b*t**p * np.pi + 2*np.pi*j/K ) + c*rand ).reshape(N,1)
        x2 = (a*t**p * np.cos( 2*b*t**p * np.pi + 2*np.pi*j/K ) + c*rand ).reshape(N,1)
        X[N*j:N*(j+1)] = np.concatenate((x1,x2), axis=1)
        y[N*j:N*(j+1)] = j
    
        
    np.save("spiral_train_features.npy", X)     # write train features and labels
    np.save("spiral_train_labels.npy", y)
    
    ## Create Test Set                       
    N = Ntest
    X = np.zeros((N*K,D))                       # feature matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8')            # class labels
    
    for j in range(K):
        t = rnd.uniform(size=N)
        rand = rnd.normal(size=N)
        x1 = (a*t**p * np.sin( 2*b*t**p * np.pi + 2*np.pi*j/K ) + c*rand ).reshape(N,1)
        x2 = (a*t**p * np.cos( 2*b*t**p * np.pi + 2*np.pi*j/K ) + c*rand ).reshape(N,1)
        X[N*j:N*(j+1)] = np.concatenate((x1,x2), axis=1)
        y[N*j:N*(j+1)] = j
        
    np.save("spiral_test_features.npy", X)      # write train features and labels
    np.save("spiral_test_labels.npy", y)
