import numpy as np
import multiprocessing as mp
from functools import partial

def string_kernel_DP_triangular_numbers_(x,y):
    z = x==y # O(M)
    tri, K = 0, 0
    for equal in z: # O(M)
        if equal: # (1)
            tri += 1 # O(1)
            K += tri # O(1)
        else:
            tri = 0 # O(1)
    return K

def string_kernel_DP_triangular_numbers_vectorized(x,Y):
    z = x==Y
    tri = np.zeros(len(z), dtype=int)
    K = np.zeros(len(z), dtype=int)
    for equal in z.T:
        tri += equal
        tri[~equal] = 0 
        K += tri
    return K

def string_kernel_DP_triangular_numbers(X,Y):
    return np.array([string_kernel_DP_triangular_numbers_vectorized(x,Y) for x in X])

def string_kernel_DP_triangular_numbers_multithread(X,Y,n_jobs=None):

    with mp.Pool(n_jobs) as pool:
        K_list = pool.map(partial(string_kernel_DP_triangular_numbers_vectorized, Y=Y), X)
    
    K = np.array(K_list).squeeze()
    
    return K

## ------------------------------- Polynomial String Kernel ------------------------------- ##

def poly_kernel_(x,y,p):
    z = x==y # O(M)
    contigs = []
    counter = 0
    for equal in z: # O(M)
        if equal: # (1)
            counter += 1 # O(1)
        else:
            contigs += [counter] # O(1)
            counter = 0 # O(1)
    contigs += [counter] # O(1)
    contigs = np.array(contigs) # O(1)
    return np.sum(contigs**p)/p # O(3M) + O(1)

def poly_kernel(X,Y,p=1.2):
    
    K = np.zeros((len(X), len(Y)), dtype=int)
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            K[i,j] = poly_kernel_(x,y,p=p)
            
    return K

def poly_kernel_multithread(X,Y,p=1.2,n_jobs=16):

    Xn, Xm = X.shape
    with mp.Pool(n_jobs) as pool:
        K_list = pool.map(partial(poly_kernel, Y=Y, p=p), X.reshape(Xn, 1, Xm))
    
    K = np.array(K_list).squeeze()
    
    return K

## ------------------------------- CovRSK ------------------------------- ##

def ohe(idx, size):
    out = np.zeros(size,dtype=int)
    out[idx-1] = 1
    return out

def CovSample(M, alpha, beta, seed=1):

    np.random.seed(seed)
    
    Ms = [1]
    for m in range(2,M+1): 
        if (1-(alpha**(m-Ms[-1]+1)))*(m**(-beta)) >= np.random.rand():
            Ms += [m]

    return Ms

def CovRSK_DP_triangular_numbers_vectorized(x,Y,Ms_ohe):
    z = x==Y
    K, tri, cov_tri = np.zeros((3, len(z)), dtype=int)
    for equal in z.T:
        tri += equal
        mask = Ms_ohe[tri] == 1
        cov_tri += mask*equal
        tri[~equal] = 0
        cov_tri[~equal] = 0
        K += cov_tri
    return K
    
def CovRSK_DP_triangular_numbers(X, Y, alpha=0.6, beta=1.0, seed=37):

    Xn, Xm = X.shape
    Ms = CovSample(Xm, alpha, beta, seed)
    Ms_ohe = ohe(idx=np.array(Ms)+1, size=Xm+1)
    K = np.array([CovRSK_DP_triangular_numbers_vectorized(x,Y,Ms_ohe=Ms_ohe) for x in X])

    return K

def CovRSK_DP_triangular_numbers_multithread(X, Y, alpha=0.6, beta=1.0, n_jobs=None, seed=37):
    
    Xn, Xm = X.shape
    Ms = CovSample(Xm, alpha, beta, seed)
    Ms_ohe = ohe(idx=np.array(Ms)+1, size=Xm+1)
    func = partial(CovRSK_DP_triangular_numbers_vectorized, Y=Y,Ms_ohe=Ms_ohe)
    with mp.Pool(n_jobs) as pool:
        K_list = pool.map(func, X)
    
    K = np.array(K_list).squeeze()
    
    return K
    
## ------------------------------- other kernels ------------------------------- ##

def linear_kernel(X,Y):
    return np.dot(X,Y.T)

def hamming_kernel(X, Y):
    return np.dot(X,Y.T) + np.dot((1-X), (1-Y).T)