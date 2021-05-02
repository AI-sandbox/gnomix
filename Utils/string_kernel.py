import numpy as np
import multiprocessing as mp
from functools import partial

slide_window = np.lib.stride_tricks.sliding_window_view

def sum_over_mz(m,z):
    return np.sum(np.all( np.lib.stride_tricks.sliding_window_view(z,m,axis=2), axis=3),axis=2)

def string_kernel_singlethread(X, Y):
    Z = np.array([np.equal(X_i, Y) for X_i in X])
    return np.sum( np.array( [m*sum_over_mz(m,Z) for m in range(Z.shape[-1])] ), axis=0 )

def string_kernel(X, Y, n_jobs=None):

    Z = np.array([np.equal(X_i, Y) for X_i in X])
        
    with mp.Pool(n_jobs) as pool:
        sums = pool.map(partial(sum_over_mz, z=Z), range(Z.shape[-1]))

    return np.sum(sums,axis=0)

def random_string_kernel_singlethread(X, Y):
    np.random.seed(seed)
    Z = np.array([np.equal(X_i, Y) for X_i in X])
    Ms = [m for m in range(1,Z.shape[-1]) if np.random.rand() < (1/m**alpha)]
    return np.sum( np.array( [m*sum_over_mz(m,Z) for m in Ms ), axis=0 )

def random_string_kernel(X, Y, alpha=1, n_jobs=None, seed=37):
    
    np.random.seed(seed)
    Z = np.array([np.equal(X_i, Y) for X_i in X])
    Ms = [m for m in range(1,Z.shape[-1]) if np.random.rand() < (1/m**alpha)]
    
    with mp.Pool(n_jobs) as pool:
        sums = pool.map(partial(sum_over_mz, z=Z), Ms)

    return np.sum(sums,axis=0)

## ------------------------------- other kernels ------------------------------- ##

def linear_kernel(X,Y):
    return np.dot(X,Y.T)

def hamming_kernel(X, Y):
    return np.dot(X,Y.T) + np.dot((1-X), (1-Y).T)

def string_kernel_singly_vectorized(X, Y):
    # From early stages of development - could be useful for 
    # different types of parallelizations
    nx, mx = X.shape
    ny, my = Y.shape
    M = mx

    K = np.zeros( (nx,ny) )
    for i, X_i in enumerate(X):
        Z_i = np.equal(X_i, Y).astype(int)
        for m in range(M):
            K[i,:] += np.sum(np.all(slide_window(Z_i,m,axis=1), axis=2),axis=1)

    return K

def substring_kernel_vectorized(X, Y, M=5, stride=1):
        
    nx, mx = X.shape
    ny, my = Y.shape

    K = np.zeros( (nx,ny) )
    for i, X_i in enumerate(X):
        Z_i = np.equal(X_i, Y).astype(int)
        C = slide_window(Z_i,M,axis=1)[:,::stride]

        for c in range(C.shape[1]):
            for m in range(M):
                K[i,:] += np.sum(np.all(slide_window(C[:,c],m,axis=1), axis=2),axis=1)

    return K