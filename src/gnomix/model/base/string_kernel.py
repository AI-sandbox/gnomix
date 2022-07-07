from dataclasses import dataclass
import numpy as np
import multiprocessing as mp
from functools import partial
from sklearn import svm

from gnomix.model.base.basemodel import SingleBase

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

@dataclass
class CovRSKBaseConfig:
    window_size:int

class CovRSKBase(SingleBase):

    def __init__(self, config: CovRSKBaseConfig):

        assert int(np.__version__.split(".")[1]) >= 20, "String kernel implementation requires numpy versions 1.20+"
        
        if config.window_size < 500:
            self.base_multithread = True
            self.kernel = CovRSK_DP_triangular_numbers
        else:
            # More robust to memory
            self.base_multithread = False
            self.kernel = CovRSK_DP_triangular_numbers_multithread


    def model_factory(self):
        return svm.SVC(kernel=self.kernel, probability=True)

    def is_multithreaded(self):
        return self.base_multithread
