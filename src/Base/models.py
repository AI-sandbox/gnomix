import numpy as np

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import svm

from src.Base.base import Base
from src.Base.string_kernel import CovRSK, CovRSK_singlethread, string_kernel, string_kernel_singlethread

class LogisticRegressionBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_multithread = True

        self.init_base_models(
            lambda : LogisticRegression(penalty="l2", C = 3., solver="liblinear", max_iter=1000)
        )


class XGBBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init_base_models(
            lambda : XGBClassifier(
                n_estimators=20, max_depth=4, learning_rate=0.1, reg_lambda=1, reg_alpha=0,
                thread=self.n_jobs, missing=self.missing_encoding, random_state=self.seed)
        )

class LGBMBase(Base):

    def __init__(self,  *args, **kwargs):
        
        from lightgbm import LGBMClassifier # This is to avoid requiring installation of lightgbm

        super().__init__(*args, **kwargs)

        self.init_base_models(
            lambda : LGBMClassifier(
                n_estimators=20, max_depth=4, learning_rate=0.1, reg_lambda=1, reg_alpha=0,
                n_jobs=self.n_jobs, random_state=self.seed) # use np.nan for missing encoding
        )

class RFBase(Base):

    def __init__(self,  *args, **kwargs):

        from sklearn.ensemble import RandomForestClassifier

        super().__init__(*args, **kwargs)

        self.init_base_models(
            lambda : RandomForestClassifier(n_estimators=20,max_depth=4,n_jobs=self.n_jobs) 
        )

class KNNBase(Base):

    from sklearn.neighbors import KNeighborsClassifier

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init_base_models(
            lambda : KNeighborsClassifier(n_neighbors=1)
        )

class SVMBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init_base_models(
            lambda : svm.SVC(C=100., gamma=0.001, probability=True)
        )

class StringKernelBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert int(np.__version__.split(".")[1]) >= 20, "String kernel implementation requires numpy versions 1.20+"

        self.log_inference = True # display progress of predict proba
        self.train_admix = False # save computation
        self.base_multithread = True
        self.kernel = string_kernel_singlethread if self.n_jobs==1 or self.base_multithread else string_kernel

        self.init_base_models(
            lambda : svm.SVC(kernel=self.kernel, probability=True)
        )

class CovRSKBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert int(np.__version__.split(".")[1]) >= 20, "String kernel implementation requires numpy versions 1.20+"
        
        self.log_inference = True # display progress of predict proba
        self.train_admix = False # save computation

        if self.M < 500:
            self.base_multithread = True
            self.kernel = CovRSK_singlethread
        else:
            # More robust to memory
            self.base_multithread = False
            self.kernel = CovRSK

        self.init_base_models(
            lambda : svm.SVC(kernel=self.kernel, probability=True)
        )