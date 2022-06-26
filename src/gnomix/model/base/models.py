import numpy as np

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import svm

from gnomix.model.base.base import Base
from gnomix.model.base.string_kernel import CovRSK_DP_triangular_numbers, CovRSK_DP_triangular_numbers_multithread
from gnomix.model.base.string_kernel import string_kernel_DP_triangular_numbers, string_kernel_DP_triangular_numbers_multithread
from gnomix.model.base.string_kernel import poly_kernel, poly_kernel_multithread

class LogisticRegressionBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_multithread = True

        self.init_base_models(
            model_factory=lambda: LogisticRegression(penalty="l2", C = 3., solver="liblinear", max_iter=1000)
        )


class XGBBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_multithread = True
        n_jobs = self.n_jobs if not self.base_multithread else 1

        self.init_base_models(
            model_factory=lambda: XGBClassifier(
                n_estimators=20, max_depth=4, learning_rate=0.1, reg_lambda=1, reg_alpha=0,
                nthread=n_jobs, missing=self.missing_encoding, random_state=self.seed)
        )

class LGBMBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        from lightgbm import LGBMClassifier 

        self.base_multithread = True
        n_jobs = self.n_jobs if not self.base_multithread else 1

        self.init_base_models(
            model_factory=lambda: LGBMClassifier(
                n_estimators=20, max_depth=4, learning_rate=0.1, reg_lambda=1, reg_alpha=0,
                n_jobs=n_jobs, random_state=self.seed) # use np.nan for missing encoding
        )

class RFBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        from sklearn.ensemble import RandomForestClassifier

        self.base_multithread = True
        n_jobs = self.n_jobs if not self.base_multithread else 1

        self.init_base_models(
            model_factory=lambda: RandomForestClassifier(n_estimators=20,max_depth=4,n_jobs=n_jobs) 
        )

class CBBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        import catboost

        self.base_multithread = True
        n_jobs = self.n_jobs if not self.base_multithread else 1

        self.init_base_models(
            model_factory=lambda: catboost.CatBoostClassifier(n_estimators=20, max_depth=4, reg_lambda=1,
                thread_count=n_jobs, verbose=0)
        )

class LDABase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        self.base_multithread = True

        self.init_base_models(
            model_factory=lambda: LinearDiscriminantAnalysis()
        )

class NBGaussianBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        from sklearn.naive_bayes import GaussianNB
        
        self.base_multithread = True

        self.init_base_models(
            model_factory=lambda: GaussianNB()
        )

class NBBernoulliBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        from sklearn.naive_bayes import BernoulliNB
        
        self.base_multithread = True

        self.init_base_models(
            model_factory=lambda: BernoulliNB(alpha=0)
        )

class NBMultinomialBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        from sklearn.naive_bayes import MultinomialNB
        
        self.base_multithread = True

        self.init_base_models(
            model_factory=lambda: MultinomialNB(alpha=0)
        )

class KNNBase(Base):
    
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        from sklearn.neighbors import KNeighborsClassifier

        self.base_multithread = True

        self.init_base_models(
            model_factory=lambda: KNeighborsClassifier(n_neighbors=1)
        )

class SVMBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log_inference = True # display progress of predict proba
        self.train_admix = False # save computation
        self.base_multithread = True

        self.init_base_models(
            model_factory=lambda: svm.SVC(C=100., gamma=0.001, probability=True)
        )

class StringKernelBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert int(np.__version__.split(".")[1]) >= 20, "String kernel implementation requires numpy versions 1.20+"

        self.log_inference = True # display progress of predict proba
        self.train_admix = False # save computation
        self.base_multithread = False # this multithreads along the base models instead of with in each window
        # self.kernel = string_kernel_singlethread if self.n_jobs==1 or self.base_multithread else string_kernel
        self.kernel = string_kernel_DP_triangular_numbers if self.n_jobs==1 or self.base_multithread else string_kernel_DP_triangular_numbers_multithread

        self.init_base_models(
            model_factory=lambda: svm.SVC(kernel=self.kernel, probability=True)
        )

class PolynomialStringKernelBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert int(np.__version__.split(".")[1]) >= 20, "String kernel implementation requires numpy versions 1.20+"

        self.log_inference = True # display progress of predict proba
        self.train_admix = False # save computation
        self.base_multithread = False # this multithreads along the base models instead of with in each window
        # self.kernel = string_kernel_singlethread if self.n_jobs==1 or self.base_multithread else string_kernel
        self.kernel = poly_kernel if self.n_jobs==1 or self.base_multithread else poly_kernel_multithread

        self.init_base_models(
            model_factory=lambda: svm.SVC(kernel=self.kernel, probability=True)
        )

class CovRSKBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert int(np.__version__.split(".")[1]) >= 20, "String kernel implementation requires numpy versions 1.20+"
        
        self.log_inference = True # display progress of predict proba
        self.train_admix = False # save computation

        if self.M < 500:
            self.base_multithread = True
            self.kernel = CovRSK_DP_triangular_numbers
        else:
            # More robust to memory
            self.base_multithread = False
            self.kernel = CovRSK_DP_triangular_numbers_multithread

        self.init_base_models(
            model_factory=lambda: svm.SVC(kernel=self.kernel, probability=True)
        )