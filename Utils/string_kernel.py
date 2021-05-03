import numpy as np
import multiprocessing as mp
from functools import partial

slide_window = np.lib.stride_tricks.sliding_window_view

def sum_over_mZ(m,Z):
    return np.sum(np.all( np.lib.stride_tricks.sliding_window_view(Z,m,axis=2), axis=3),axis=2)

def sum_over_zM(z, Ms):
    return np.sum([np.sum(np.all(np.lib.stride_tricks.sliding_window_view(z,m,axis=1), axis=2),axis=1) for m in Ms], axis=0)

def string_kernel(X, Y, n_jobs=None, m_axis="samples"):
    
    Z = np.array([np.equal(X_i, Y) for X_i in X])
    Ms = range(1,Z.shape[-1])
    
    if n_jobs == 1 or m_axis is "None":
        K = np.sum( np.array( [sum_over_mZ(m,Z) for m in Ms] ), axis=0 )

    elif m_axis == "k-mers":
        with mp.Pool(n_jobs) as pool:
            sums = pool.map(partial(sum_over_mZ, Z=Z), Ms)
        K = np.sum(sums,axis=0)

    elif m_axis == "samples":
        with mp.Pool(n_jobs) as pool:
            K = np.array(pool.map(partial(sum_over_zM, Ms=Ms),  Z))

    return K

def random_string_kernel(X, Y, alpha=1, n_jobs=None, m_axis="samples", seed=37):
    
    np.random.seed(seed)
    Z = np.array([np.equal(X_i, Y) for X_i in X])
    Ms = [m for m in range(1,Z.shape[-1]) if np.random.rand() <= (1/m**alpha)]
    
    if n_jobs == 1 or m_axis is "None":
        K = np.sum( np.array( [sum_over_mZ(m,Z) for m in Ms] ), axis=0 )

    elif m_axis == "k-mers":
        with mp.Pool(n_jobs) as pool:
            sums = pool.map(partial(sum_over_mZ, Z=Z), Ms)
        K = np.sum(sums,axis=0)

    elif m_axis == "samples":
        with mp.Pool(n_jobs) as pool:
            K = np.array(pool.map(partial(sum_over_zM, Ms=Ms),  Z))

    return K

def string_kernel_singlethread(X, Y):
    Z = np.array([np.equal(X_i, Y) for X_i in X])
    return np.sum( np.array( [m*sum_over_mZ(m,Z) for m in range(Z.shape[-1])] ), axis=0 )
    
def random_string_kernel_singlethread(X, Y, alpha=1, seed=37):
    np.random.seed(seed)
    Z = np.array([np.equal(X_i, Y) for X_i in X])
    Ms = [m for m in range(1,Z.shape[-1]) if np.random.rand() < (1/m**alpha)]
    return np.sum( np.array( [sum_over_mZ(m,Z) for m in Ms] ), axis=0 )

## ------------------------------- other kernels ------------------------------- ##

def linear_kernel(X,Y):
    return np.dot(X,Y.T)

def hamming_kernel(X, Y):
    return np.dot(X,Y.T) + np.dot((1-X), (1-Y).T)

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