import numpy as np
import multiprocessing as mp
from functools import partial

def get_slide_window():
    try:
        return np.lib.stride_tricks.sliding_window_view 
    except AttributeError:
        print("Error: String kernel implementation requires numpy versions 1.20+")

def sum_over_mZ_loopy(m,Z):
    slide_window = get_slide_window()
    return np.array( [np.sum(np.all( slide_window(z,m,axis=1), axis=2),axis=1) for z in Z ] , dtype=np.int16)

def sum_over_mZ(m,Z):

    # avoid memory crash
    if np.prod(Z.shape)*m > (1500*1500*2000)*100:
        return sum_over_mZ_loopy(m,Z)
    
    slide_window = get_slide_window()

    return np.sum(np.all( slide_window(Z,m,axis=2), axis=3),axis=2,dtype=np.int16)

def sum_over_zM(z, Ms):
    slide_window = get_slide_window()
    return np.sum([np.sum(np.all( slide_window(z,m,axis=1), axis=2),axis=1) for m in Ms], axis=0)

def string_kernel(X, Y, K_max=None, n_jobs=None, m_axis="samples"):
    
    Z = np.array([np.equal(X_i, Y) for X_i in X])
    Ms = range(1,Z.shape[-1]) if K_max is None else range(1,K_max)
    
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
    return np.sum( np.array( [sum_over_mZ(m,Z) for m in range(Z.shape[-1])] ), axis=0 )

def CovSample(M, alpha, beta, seed=37):

    np.random.seed(seed)
    
    Ms = [1]
    for m in range(2,M+1): 
        if (1/m**beta)*(1-np.sqrt(alpha**(m-Ms[-1]))) >= np.random.rand():
            Ms += [m]

    return Ms

def CovRSK(X, Y, alpha=0.6, beta=1.0, n_jobs=None, seed=37):
    
    Z = np.array([np.equal(X_i, Y) for X_i in X])
    Ms = CovSample(Z.shape[-1], alpha, beta, seed)

    with mp.Pool(n_jobs) as pool:
        K = np.array(pool.map(partial(sum_over_zM, Ms=Ms),  Z))

    return K

def CovRSK_singlethread(X, Y, alpha=0.9, beta=0.5, seed=37):
    
    Z = np.array([np.equal(X_i, Y) for X_i in X])
    Ms = CovSample(Z.shape[-1], alpha, beta, seed)
    return np.sum( np.array( [sum_over_mZ(m,Z) for m in Ms] ), axis=0 )

    
## ------------------------------- other kernels ------------------------------- ##

def random_string_kernel_singlethread(X, Y, alpha=1, seed=37):
    np.random.seed(seed)
    Z = np.array([np.equal(X_i, Y) for X_i in X])
    Ms = [m for m in range(1,Z.shape[-1]) if np.random.rand() < (1/m**alpha)]
    return np.sum( np.array( [sum_over_mZ(m,Z) for m in Ms] ), axis=0 )

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

def linear_kernel(X,Y):
    return np.dot(X,Y.T)

def hamming_kernel(X, Y):
    return np.dot(X,Y.T) + np.dot((1-X), (1-Y).T)

def substring_kernel_vectorized(X, Y, M=5, stride=1):

    slide_window = get_slide_window()

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